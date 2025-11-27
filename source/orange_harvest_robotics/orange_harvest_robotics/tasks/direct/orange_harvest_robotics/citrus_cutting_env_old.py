"""Citrus Cutting Environment - IK Control with Vision Pipeline"""

import torch
import numpy as np
from collections.abc import Sequence
from enum import IntEnum

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, ContactSensor, FrameTransformer
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
    sample_uniform,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController

from .citrus_cutting_env_cfg import CitrusCuttingEnvCfg


class CuttingPhase(IntEnum):
    """States for the cutting state machine."""
    IDLE = 0
    DETECTING = 1
    APPROACHING = 2
    ALIGNING = 3
    CLOSING_GRIPPER = 4
    WAITING = 5
    OPENING_GRIPPER = 6
    DONE = 7


class CitrusCuttingEnv(DirectRLEnv):
    """Environment for robotic citrus cutting using IK control and vision.
    
    State machine:
    1. IDLE -> DETECTING: Reset, open gripper
    2. DETECTING: Run vision pipeline to find orange/stem
    3. APPROACHING: Move to approach position (above target)
    4. ALIGNING: Fine-tune position using hand camera (optional)
    5. CLOSING_GRIPPER: Close gripper to cut
    6. WAITING: Wait for cutting to complete
    7. OPENING_GRIPPER: Open gripper
    8. DONE: Episode complete
    """

    cfg: CitrusCuttingEnvCfg

    def __init__(self, cfg: CitrusCuttingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get robot joint indices
        self.arm_joint_ids, _ = self.robot.find_joints("panda_joint.*")
        self.gripper_joint_ids, _ = self.robot.find_joints("panda_finger.*")
        self.num_arm_joints = len(self.arm_joint_ids)

        # Get end-effector body index
        self.ee_body_id, _ = self.robot.find_bodies("panda_hand")
        self.ee_body_id = self.ee_body_id[0]

        # Setup IK controller
        self._setup_ik_controller()

        # Buffers for target pose (computed from vision)
        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)  # In robot base frame
        self.target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat_b[:, 0] = 1.0  # Identity quaternion

        # State machine
        self.cutting_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_timer = torch.zeros(self.num_envs, device=self.device)

        # Success tracking
        self.cut_successful = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stem_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Gripper state
        self.gripper_open = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Create visualization markers
        self._create_target_markers()

        # Compute camera intrinsics matrices
        self._compute_camera_intrinsics()

        # Set observation space
        self.cfg.observation_space = (
            self.num_arm_joints * 2  # joint pos + vel
            + 7  # ee pose in base frame
            + 7  # target pose in base frame
            + 7  # relative pose (ee to target)
            + 1  # gripper state
            + 1  # phase
        )

    def _setup_scene(self):
        """Setup the scene with all entities."""
        # Create robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Create tree objects (trunk + branch instead of single tree)
        self.tree_trunk = RigidObject(self.cfg.tree_trunk_cfg)
        self.tree_branch = RigidObject(self.cfg.tree_branch_cfg)
        self.orange = RigidObject(self.cfg.orange_cfg)
        self.stem = RigidObject(self.cfg.stem_cfg)
        
        # Create sensors
        self.fixed_camera = Camera(self.cfg.fixed_camera_cfg)
        self.hand_camera = Camera(self.cfg.hand_camera_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)

        # Add ground plane and lighting
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Add lights
        from isaaclab.sim.spawners.lights import spawn_light
        spawn_light(
            prim_path="/World/DomeLight",
            cfg=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
        )
        spawn_light(
            prim_path="/World/DistantLight", 
            cfg=sim_utils.DistantLightCfg(intensity=600.0, angle=0.53, color=(1.0, 1.0, 0.95)),
        )

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Add articulations to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add rigid objects to scene
        self.scene.rigid_objects["tree_trunk"] = self.tree_trunk
        self.scene.rigid_objects["tree_branch"] = self.tree_branch
        self.scene.rigid_objects["orange"] = self.orange
        self.scene.rigid_objects["stem"] = self.stem
        
        # Add sensors to scene
        self.scene.sensors["fixed_camera"] = self.fixed_camera
        self.scene.sensors["hand_camera"] = self.hand_camera
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["ee_frame"] = self.ee_frame

        # Set camera view
        self.sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.5, 0.0, 0.8])

    def _setup_ik_controller(self):
        """Setup differential IK controller for Franka."""
        self.ik_controller = DifferentialIKController(
            self.cfg.ik_controller_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _create_target_markers(self):
        """Create visualization markers for target pose."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetPose",
            markers={
                "target": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            },
        )
        self.target_markers = VisualizationMarkers(marker_cfg)

    def _compute_camera_intrinsics(self):
        """Compute camera intrinsic matrices from config."""
        # Fixed camera
        fx_fixed = self._focal_length_to_fx(
            self.cfg.fixed_camera_intrinsics["focal_length"],
            self.cfg.fixed_camera_intrinsics["horizontal_aperture"],
            self.cfg.fixed_camera_intrinsics["width"]
        )
        fy_fixed = fx_fixed
        cx_fixed = self.cfg.fixed_camera_intrinsics["width"] / 2.0
        cy_fixed = self.cfg.fixed_camera_intrinsics["height"] / 2.0
        
        self.fixed_camera_K = torch.tensor([
            [fx_fixed, 0, cx_fixed],
            [0, fy_fixed, cy_fixed],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Hand camera
        fx_hand = self._focal_length_to_fx(
            self.cfg.hand_camera_intrinsics["focal_length"],
            self.cfg.hand_camera_intrinsics["horizontal_aperture"],
            self.cfg.hand_camera_intrinsics["width"]
        )
        fy_hand = fx_hand
        cx_hand = self.cfg.hand_camera_intrinsics["width"] / 2.0
        cy_hand = self.cfg.hand_camera_intrinsics["height"] / 2.0
        
        self.hand_camera_K = torch.tensor([
            [fx_hand, 0, cx_hand],
            [0, fy_hand, cy_hand],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)

    @staticmethod
    def _focal_length_to_fx(focal_length_mm: float, horizontal_aperture_mm: float, width_pixels: int) -> float:
        """Convert focal length to pixel focal length."""
        return (focal_length_mm / horizontal_aperture_mm) * width_pixels

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self):
        """Apply actions to the robot using IK controller."""
        # If robot is frozen for debugging, don't apply any commands
        if hasattr(self.cfg, 'freeze_robot') and self.cfg.freeze_robot:
            return
        
        if (self.cutting_phase == CuttingPhase.APPROACHING).any() or \
           (self.cutting_phase == CuttingPhase.ALIGNING).any():
            # Apply IK to reach target
            self._apply_ik_control()
        
        # Apply gripper command
        self._apply_gripper_control()

    def _apply_ik_control(self):
        """Compute and apply IK joint targets."""
        # Get current joint positions
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        
        # Get current end-effector pose in base frame (from frame transformer)
        ee_pos_b = self.ee_frame.data.target_pos_source[:, 0, :]
        ee_quat_b = self.ee_frame.data.target_quat_source[:, 0, :]
        
        # Get Jacobian
        jacobian = self.robot.root_physx_view.get_jacobians()
        ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        ee_jacobian = jacobian[:, ee_jacobi_idx, :, self.arm_joint_ids]
        
        # Set IK command (target pose in base frame)
        self.ik_controller.set_command(
            torch.cat([self.target_pos_b, self.target_quat_b], dim=-1)
        )
        
        # Compute desired joint positions
        joint_pos_des = self.ik_controller.compute(
            ee_pos_b, ee_quat_b, ee_jacobian, joint_pos
        )
        
        # Apply joint position targets
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.arm_joint_ids)

    def _apply_gripper_control(self):
        """Apply binary gripper control."""
        gripper_targets = torch.where(
            self.gripper_open.unsqueeze(-1),
            torch.tensor([[0.04, 0.04]], device=self.device),  # Open
            torch.tensor([[0.0, 0.0]], device=self.device),    # Closed
        )
        self.robot.set_joint_position_target(gripper_targets, joint_ids=self.gripper_joint_ids)

    def _get_observations(self) -> dict:
        """Compute observations."""
        # Joint state
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        
        # End-effector pose in base frame
        ee_pos_b = self.ee_frame.data.target_pos_source[:, 0, :]
        ee_quat_b = self.ee_frame.data.target_quat_source[:, 0, :]
        
        # Relative pose (ee to target)
        rel_pos, rel_quat = subtract_frame_transforms(
            ee_pos_b, ee_quat_b,
            self.target_pos_b, self.target_quat_b,
        )
        
        # Concatenate observations
        obs = torch.cat(
            [
                joint_pos,
                joint_vel,
                ee_pos_b,
                ee_quat_b,
                self.target_pos_b,
                self.target_quat_b,
                rel_pos,
                rel_quat,
                self.gripper_open.unsqueeze(-1).float(),
                self.cutting_phase.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on phase and success."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Reward for detecting orange
        rewards += self.stem_detected.float() * 1.0
        
        # Reward for reaching target
        ee_pos_b = self.ee_frame.data.target_pos_source[:, 0, :]
        dist_to_target = torch.norm(ee_pos_b - self.target_pos_b, dim=-1)
        rewards += -dist_to_target * 0.5
        
        # Large reward for successful cut
        rewards += self.cut_successful.float() * 100.0
        
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.cut_successful.clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Reset robot to home position
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        # Small randomization
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, self.device)
        
        # Reset root state (add environment origins!)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset orange and stem with randomization
        self._reset_orange_and_stem(env_ids)
        
        # Reset state machine
        self.cutting_phase[env_ids] = CuttingPhase.IDLE
        self.phase_timer[env_ids] = 0.0
        self.cut_successful[env_ids] = False
        self.stem_detected[env_ids] = False
        self.gripper_open[env_ids] = True
        
        # Clear target
        self.target_pos_b[env_ids] = 0.0
        self.target_quat_b[env_ids] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)

    def _reset_orange_and_stem(self, env_ids: torch.Tensor):
        """Reset orange and stem positions with randomization."""
        num_resets = len(env_ids)
        
        # Randomize orange position (around branch end)
        orange_pos = self.orange.data.default_root_state[env_ids, :3].clone()
        orange_pos[:, 0] += sample_uniform(-0.05, 0.05, (num_resets,), self.device)
        orange_pos[:, 1] += sample_uniform(-0.05, 0.05, (num_resets,), self.device)
        orange_pos[:, 2] += sample_uniform(-0.02, 0.02, (num_resets,), self.device)
        orange_pos += self.scene.env_origins[env_ids]
        
        # Stem position (between branch end and orange)
        branch_end_pos = torch.tensor([0.6, -0.3, 1.0], device=self.device).unsqueeze(0).expand(num_resets, -1)
        branch_end_pos += self.scene.env_origins[env_ids]
        stem_pos = (orange_pos + branch_end_pos) / 2.0
        
        # Write to simulation
        orange_state = self.orange.data.default_root_state[env_ids].clone()
        orange_state[:, :3] = orange_pos
        self.orange.write_root_state_to_sim(orange_state, env_ids)
        
        stem_state = self.stem.data.default_root_state[env_ids].clone()
        stem_state[:, :3] = stem_pos
        self.stem.write_root_state_to_sim(stem_state, env_ids)

    def _update_state_machine(self):
        """Update state machine for cutting sequence."""
        dt = self.sim.get_physics_dt() * self.cfg.decimation
        
        for env_id in range(self.num_envs):
            phase = self.cutting_phase[env_id].item()
            
            if phase == CuttingPhase.IDLE:
                self.cutting_phase[env_id] = CuttingPhase.DETECTING
                self.gripper_open[env_id] = True
                
            elif phase == CuttingPhase.DETECTING:
                success = self._detect_target(env_id)
                if success:
                    self.stem_detected[env_id] = True
                    self.cutting_phase[env_id] = CuttingPhase.APPROACHING
                    
            elif phase == CuttingPhase.APPROACHING:
                ee_pos = self.ee_frame.data.target_pos_source[env_id, 0, :]
                dist = torch.norm(ee_pos - self.target_pos_b[env_id])
                if dist < self.cfg.target_position_tolerance:
                    self.cutting_phase[env_id] = CuttingPhase.ALIGNING
                    
            elif phase == CuttingPhase.ALIGNING:
                self.cutting_phase[env_id] = CuttingPhase.CLOSING_GRIPPER
                self.gripper_open[env_id] = False
                self.phase_timer[env_id] = 0.0
                
            elif phase == CuttingPhase.CLOSING_GRIPPER:
                self.phase_timer[env_id] += dt
                if self.phase_timer[env_id] > self.cfg.cutting_wait_time:
                    self.cutting_phase[env_id] = CuttingPhase.WAITING
                    self.phase_timer[env_id] = 0.0
                    
            elif phase == CuttingPhase.WAITING:
                self.phase_timer[env_id] += dt
                if self.phase_timer[env_id] > self.cfg.cutting_wait_time:
                    self.cut_successful[env_id] = True
                    self.cutting_phase[env_id] = CuttingPhase.OPENING_GRIPPER
                    self.gripper_open[env_id] = True
                    
            elif phase == CuttingPhase.OPENING_GRIPPER:
                self.cutting_phase[env_id] = CuttingPhase.DONE

    def _detect_target(self, env_id: int) -> bool:
        """Run vision pipeline to detect orange/stem and compute target pose."""
        # Get camera data
        rgb_image = self.fixed_camera.data.output["rgb"][env_id]
        depth_image = self.fixed_camera.data.output["distance_to_image_plane"][env_id]
        
        # Use ground truth orange position
        orange_pos_w = self.orange.data.root_pos_w[env_id]
        
        # Get robot base pose
        robot_base_pos_w = self.robot.data.root_pos_w[env_id]
        robot_base_quat_w = self.robot.data.root_quat_w[env_id]
        
        # Transform target from world to robot base frame
        target_pos_b, _ = subtract_frame_transforms(
            robot_base_pos_w, robot_base_quat_w,
            orange_pos_w, torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device),
        )
        
        # Set target position (top of orange + approach offset)
        self.target_pos_b[env_id] = target_pos_b
        self.target_pos_b[env_id, 2] += 0.04 + self.cfg.approach_height_offset
        
        # Set target orientation (gripper facing down)
        self.target_quat_b[env_id] = torch.tensor(
            [0.7071, 0.0, -0.7071, 0.0], dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(1, -1)[0]
        
        # Visualize target
        target_pos_w = self._transform_base_to_world(
            self.target_pos_b[env_id],
            robot_base_pos_w, robot_base_quat_w
        )
        target_quat_w = quat_mul(robot_base_quat_w, self.target_quat_b[env_id])
        
        self.target_markers.visualize(
            target_pos_w.unsqueeze(0),
            target_quat_w.unsqueeze(0),
        )
        
        return True

    def _transform_world_to_camera(self, point_w: torch.Tensor, camera_pose_w: tuple) -> torch.Tensor:
        """Transform 3D point from world frame to camera frame."""
        cam_pos_w, cam_quat_w = camera_pose_w
        point_cam = point_w - cam_pos_w
        cam_quat_inv = quat_inv(cam_quat_w)
        point_cam = quat_apply(cam_quat_inv, point_cam)
        return point_cam

    def _transform_base_to_world(self, point_b: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
        """Transform 3D point from robot base frame to world frame."""
        point_w = quat_apply(base_quat_w, point_b)
        point_w += base_pos_w
        return point_w

    def _project_depth_to_3d(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Project 2D pixel coordinates + depth to 3D points in camera frame."""
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points_3d = torch.stack([x, y, z], dim=-1)
        return points_3d

    def step(self, actions: torch.Tensor) -> tuple:
        """Override step to include state machine update."""
        self._update_state_machine()
        return super().step(actions)