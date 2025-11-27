# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


@configclass
class CitrusCuttingEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.0  # 480 timesteps at 60Hz
    decimation = 2
    action_space = 9  # 7 arm joints + 2 finger joints
    observation_space = 36  # Updated to match actual observations
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Tree trunk (vertical cylinder at base)
    tree_trunk_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TreeTrunk",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.5)),
    )

    # Tree branch (horizontal)
    tree_branch_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TreeBranch",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.15, 1.0),
            rot=(0.7071, 0.7071, 0.0, 0.0),  # Horizontal
        ),
    )

    # Stem (vertical, hangs from branch)
    stem_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Stem",
        spawn=sim_utils.CylinderCfg(
            radius=0.005,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.002),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.4, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.3, 0.925),
        ),
    )

    # Orange (hangs from stem)
    orange_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Orange",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.3, 0.81),
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 2.0
    rot_reward_scale = 1.5
    grasp_reward_scale = 5.0
    cut_reward_scale = 10.0
    action_penalty_scale = 0.01
    finger_reward_scale = 2.0


class CitrusCuttingEnv(DirectRLEnv):
    cfg: CitrusCuttingEnvCfg

    def __init__(self, cfg: CitrusCuttingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Robot DOF limits and speed scales
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Get local grasp poses from USD stage
        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        # Compute gripper center grasp pose
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # Stem local grasp pose (center of stem, gripper approaches from side)
        stem_local_grasp_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.stem_local_grasp_pos = stem_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.stem_local_grasp_rot = stem_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # Gripper and stem axes for alignment rewards
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.stem_perpendicular_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.stem_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        # Get body indices
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        
        # Get finger joint index as integer for cleaner indexing
        finger_joint_ids = self._robot.find_joints("panda_finger_joint1")[0]
        if isinstance(finger_joint_ids, torch.Tensor):
            self.finger_joint_idx = finger_joint_ids[0].item() if finger_joint_ids.numel() > 0 else finger_joint_ids.item()
        else:
            self.finger_joint_idx = finger_joint_ids[0] if isinstance(finger_joint_ids, list) else finger_joint_ids

        # Buffers for grasp transforms
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.stem_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.stem_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Success tracking
        self.cut_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._tree_trunk = RigidObject(self.cfg.tree_trunk_cfg)
        self._tree_branch = RigidObject(self.cfg.tree_branch_cfg)
        self._stem = RigidObject(self.cfg.stem_cfg)
        self._orange = RigidObject(self.cfg.orange_cfg)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["tree_trunk"] = self._tree_trunk
        self.scene.rigid_objects["tree_branch"] = self._tree_branch
        self.scene.rigid_objects["stem"] = self._stem
        self.scene.rigid_objects["orange"] = self._orange

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Success condition: gripper closed with sufficient force on stem
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.cut_success.clone()
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh intermediate values
        self._compute_intermediate_values()
        
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._robot.data.joint_pos,
            self.robot_grasp_pos,
            self.stem_grasp_pos,
            self.robot_grasp_rot,
            self.stem_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.stem_perpendicular_axis,
            self.gripper_up_axis,
            self.stem_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self.cfg.cut_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        # Robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset objects with randomization
        self._reset_objects(env_ids)

        # Reset success flag
        self.cut_success[env_ids] = False

        # Refresh intermediate values
        self._compute_intermediate_values(env_ids)

    def _reset_objects(self, env_ids: torch.Tensor):
        """Reset tree objects with randomization."""
        num_resets = len(env_ids)
        
        # Randomize stem and orange position slightly
        stem_pos = self._stem.data.default_root_state[env_ids, :3].clone()
        stem_pos[:, 0] += sample_uniform(-0.05, 0.05, (num_resets,), self.device)
        stem_pos[:, 1] += sample_uniform(-0.05, 0.05, (num_resets,), self.device)
        stem_pos[:, 2] += sample_uniform(-0.02, 0.02, (num_resets,), self.device)
        
        orange_pos = self._orange.data.default_root_state[env_ids, :3].clone()
        orange_pos[:, 0] = stem_pos[:, 0]
        orange_pos[:, 1] = stem_pos[:, 1]
        orange_pos[:, 2] = stem_pos[:, 2] - 0.115  # Below stem
        
        # CRITICAL: Add environment origins so objects spawn in correct environment
        stem_pos += self.scene.env_origins[env_ids]
        orange_pos += self.scene.env_origins[env_ids]
        
        # Write states
        stem_state = self._stem.data.default_root_state[env_ids].clone()
        stem_state[:, :3] = stem_pos
        self._stem.write_root_state_to_sim(stem_state, env_ids)
        
        orange_state = self._orange.data.default_root_state[env_ids].clone()
        orange_state[:, :3] = orange_pos
        self._orange.write_root_state_to_sim(orange_state, env_ids)

    def _get_observations(self) -> dict:
        # Scale joint positions to [-1, 1]
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # Vector from gripper to stem
        to_target = self.stem_grasp_pos - self.robot_grasp_pos
        
        # Gripper width (use stored index for clean indexing)
        gripper_width = self._robot.data.joint_pos[:, self.finger_joint_idx].unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,  # 9
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 9
                to_target,  # 3
                self.stem_grasp_pos,  # 3
                self.robot_grasp_pos,  # 3
                self.stem_grasp_rot,  # 4 (quaternion)
                self.robot_grasp_rot,  # 4
                gripper_width,  # 1
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Compute grasp transforms for gripper and stem."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        stem_pos = self._stem.data.root_pos_w[env_ids]
        stem_rot = self._stem.data.root_quat_w[env_ids]
        
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.stem_grasp_rot[env_ids],
            self.stem_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            stem_rot,
            stem_pos,
            self.stem_local_grasp_rot[env_ids],
            self.stem_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        joint_positions,
        franka_grasp_pos,
        stem_grasp_pos,
        franka_grasp_rot,
        stem_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        stem_perpendicular_axis,
        gripper_up_axis,
        stem_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        grasp_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        cut_reward_scale,
    ):
        # Distance from gripper to stem center
        d = torch.norm(franka_grasp_pos - stem_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        # Orientation alignment: gripper should approach stem perpendicularly
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(stem_grasp_rot, stem_perpendicular_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(stem_grasp_rot, stem_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # Action penalty
        action_penalty = torch.sum(actions**2, dim=-1)

        # Finger alignment with stem
        stem_z = stem_grasp_pos[:, 2]
        lfinger_z = franka_lfinger_pos[:, 2]
        rfinger_z = franka_rfinger_pos[:, 2]
        
        # Penalty if fingers not aligned with stem height
        lfinger_dist = torch.abs(lfinger_z - stem_z)
        rfinger_dist = torch.abs(rfinger_z - stem_z)
        finger_dist_penalty = -(lfinger_dist + rfinger_dist)

        # Grasp reward: encourage closing gripper when near stem
        gripper_width = joint_positions[:, self.finger_joint_idx]  # Use stored index
        near_stem = d < 0.03
        grasp_reward = torch.where(
            near_stem,
            (0.04 - gripper_width) * 10.0,  # Reward closing when near
            torch.zeros_like(gripper_width),
        )

        # Cut success detection: gripper closed and at stem
        gripper_closed = gripper_width < 0.01
        at_stem = d < 0.02
        cut_success = gripper_closed & at_stem
        self.cut_success = cut_success
        cut_reward = cut_success.float()

        # Combine rewards
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + grasp_reward_scale * grasp_reward
            + finger_reward_scale * finger_dist_penalty
            + cut_reward_scale * cut_reward
            - action_penalty_scale * action_penalty
        )

        # Logging
        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "grasp_reward": (grasp_reward_scale * grasp_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "finger_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
            "cut_reward": (cut_reward_scale * cut_reward).mean(),
            "success_rate": cut_success.float().mean(),
        }

        # Bonus rewards for progress
        rewards = torch.where(d < 0.05, rewards + 0.5, rewards)  # Close to stem
        rewards = torch.where(d < 0.03, rewards + 1.0, rewards)  # Very close
        rewards = torch.where(gripper_closed & near_stem, rewards + 2.0, rewards)  # Grasping

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        stem_rot,
        stem_pos,
        stem_local_grasp_rot,
        stem_local_grasp_pos,
    ):
        """Compute global grasp poses from local grasp poses."""
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_stem_rot, global_stem_pos = tf_combine(
            stem_rot, stem_pos, stem_local_grasp_rot, stem_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_stem_rot, global_stem_pos