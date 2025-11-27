"""Configuration for Citrus Cutting Environment - Simplified IK Test"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
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
    """Configuration for the citrus cutting scene."""
    pass


##
# Environment Configuration
##

@configclass
class CitrusCuttingEnvCfg(DirectRLEnvCfg):
    """Configuration for the Citrus Cutting environment - IK test."""

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

    # Scene
    scene: CitrusSceneCfg = CitrusSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Robot configuration
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

    # Environment settings
    decimation = 4
    episode_length_s = 30.0
    
    # IK controller configuration
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    
    # Action space: [x, y, z, qw, qx, qy, qz]
    action_space = 7
    observation_space = 0  # Set in environment
    state_space = 0

    # Target pose parameters
    target_position_tolerance = 0.02  # 2cm
    target_orientation_tolerance = 0.15  # ~8.6 degrees
    
    # Debug options
    freeze_robot = False  # Set to True to freeze robot

    def __post_init__(self):
        """Post initialization."""
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