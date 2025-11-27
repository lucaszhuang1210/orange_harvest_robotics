"""Configuration for Citrus Cutting Environment - IK Control with Franka Panda"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg

# Import Franka Panda configuration
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

##
# Scene Configuration
##

@configclass
class CitrusSceneCfg(InteractiveSceneCfg):
    """Configuration for the citrus cutting scene.
    
    For Direct workflow, only include static/non-interactive elements here.
    Robot, objects, sensors are defined separately in EnvCfg and spawned in _setup_scene.
    """
    pass


##
# Environment Configuration
##

@configclass
class CitrusCuttingEnvCfg(DirectRLEnvCfg):
    """Configuration for the Citrus Cutting environment with IK control."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene (just container, no entities yet)
    scene: CitrusSceneCfg = CitrusSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Robot configuration (spawned in _setup_scene)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # Frame transformer for end-effector
    ee_frame_cfg: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
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
                diffuse_color=(0.4, 0.25, 0.1),  # Brown
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.5)),
    )

    # Tree branch (horizontal, extends from trunk)
    tree_branch_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TreeBranch",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.3,  # 30cm horizontal branch
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.1),  # Brown
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.15, 1.0),  # At top of trunk, extends left
            rot=(0.7071, 0.7071, 0.0, 0.0),  # Rotate 90deg to be horizontal (Y-axis rotation)
        ),
    )

    # Stem (vertical, hangs from branch)
    stem_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Stem",
        spawn=sim_utils.CylinderCfg(
            radius=0.005,  # 1cm diameter
            height=0.15,  # 15cm long (increased from 5cm)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Make kinematic so it doesn't fall
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.002),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.4, 0.0),  # Green
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.3, 0.925),  # Hangs from end of branch (1.0 - 0.15/2 = 0.925)
        ),
    )

    # Orange (hangs from stem)
    orange_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Orange",
        spawn=sim_utils.SphereCfg(
            radius=0.04,  # 8cm diameter orange
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Make kinematic so it doesn't fall
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # 150g
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),  # Orange color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.3, 0.81),  # Below stem (0.925 - 0.15/2 - 0.04 = 0.81)
        ),
    )

    # Fixed camera configuration
    fixed_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/FixedCamera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.3, 0.0, 1.5),  # Moved to opposite side
            rot=(0.9238796, 0.0, 0.3826834, 0.0),  # Looking at tree from angle
            convention="world",
        ),
    )

    # Hand camera configuration
    hand_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/HandCamera",
        update_period=0.033,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=100.0,
            horizontal_aperture=15.0,
            clipping_range=(0.01, 1.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Contact sensor configuration
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_.*finger",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

    # Environment settings
    decimation = 4
    episode_length_s = 30.0
    
    # IK controller configuration
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    
    # Action space: [x, y, z, qw, qx, qy, qz, gripper_open]
    action_space = 8
    observation_space = 0  # Set in environment
    state_space = 0

    # Camera intrinsics
    fixed_camera_intrinsics = {
        "width": 640,
        "height": 480,
        "focal_length": 24.0,
        "horizontal_aperture": 20.955,
    }
    
    hand_camera_intrinsics = {
        "width": 320,
        "height": 240,
        "focal_length": 18.0,
        "horizontal_aperture": 15.0,
    }

    # Cutting sequence parameters
    approach_height_offset = 0.10  # Approach 10cm above target
    cutting_wait_time = 2.0  # Wait 2 seconds for gripper to close/cut
    gripper_close_threshold = 0.01  # Consider closed if joints < 0.01m

    # Target pose parameters
    target_position_tolerance = 0.02  # 2cm
    target_orientation_tolerance = 0.15  # ~8.6 degrees
    
    # Debug options
    freeze_robot = True  # Set to True to freeze robot for debugging camera positions

    def __post_init__(self):
        """Post initialization."""
        # Enable contact sensors for the robot
        self.robot_cfg.spawn.activate_contact_sensors = True
        
        # Compute max episode length
        self.max_episode_length = math.ceil(self.episode_length_s / (self.sim.dt * self.decimation))
        return super().__post_init__()


@configclass
class CitrusCuttingEnvCfg_PLAY(CitrusCuttingEnvCfg):
    """Configuration for visualization/testing."""
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.ee_frame_cfg.debug_vis = True
        self.contact_sensor_cfg.debug_vis = True