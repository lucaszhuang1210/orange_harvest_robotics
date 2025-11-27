#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test citrus cutting environment without training.

This script helps debug the environment by:
1. Creating the environment
2. Running random actions
3. Printing observations and rewards
4. Checking for errors

Usage:
    python test_env.py
    python test_env.py --num_envs 1  # Single environment for detailed output
    python test_env.py --num_steps 1000  # Run for longer
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test citrus cutting environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to simulate.")

# append AppLauncher args (includes --device)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np

import Citrus.tasks.direct.citrus  # noqa: F401
from Citrus.tasks.direct.citrus.citrus_cutting_env import CitrusCuttingEnvCfg


def print_env_info(env):
    """Print environment information."""
    print(f"\n{'='*70}")
    print("ENVIRONMENT INFO")
    print(f"{'='*70}")
    print(f"Task: Isaac-Citrus-Cutting-Direct-v0")
    print(f"Num envs: {env.unwrapped.num_envs}")
    print(f"Device: {env.unwrapped.device}")
    print(f"Physics dt: {env.unwrapped.cfg.sim.dt:.4f}s")
    print(f"Control dt: {env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation:.4f}s")
    print(f"Max episode length: {env.unwrapped.max_episode_length} steps")
    print(f"Episode duration: {env.unwrapped.max_episode_length * env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation:.2f}s")
    
    # Observation space - check if it's a dict or Box
    if hasattr(env.observation_space, 'spaces'):
        # Dict space
        print(f"\nObservation space (dict):")
        for key, space in env.observation_space.spaces.items():
            print(f"  {key}: {space.shape}")
    else:
        # Box space - single observation
        print(f"\nObservation space: {env.observation_space.shape}")
    
    print(f"Action space: {env.action_space.shape}")
    print(f"\nRobot joints: {env.unwrapped._robot.num_joints}")
    print(f"Robot bodies: {env.unwrapped._robot.num_bodies}")
    print(f"{'='*70}\n")


def check_observations(obs):
    """Check observations for issues."""
    # Handle dict observations
    if isinstance(obs, dict):
        obs_tensor = obs["policy"]
    else:
        obs_tensor = obs
    
    has_nan = torch.isnan(obs_tensor).any()
    has_inf = torch.isinf(obs_tensor).any()
    
    if has_nan:
        print("⚠️  WARNING: Observations contain NaN values!")
        return False
    if has_inf:
        print("⚠️  WARNING: Observations contain Inf values!")
        return False
    
    return True


def print_statistics(step, obs, rewards, info):
    """Print statistics every N steps."""
    # Handle dict observations
    if isinstance(obs, dict):
        obs_tensor = obs["policy"]
    else:
        obs_tensor = obs
    
    print(f"\nStep {step}:")
    print(f"  Obs shape: {obs_tensor.shape}")
    print(f"  Obs range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
    print(f"  Reward mean: {rewards.mean():.3f}")
    print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    # Print reward components if available
    if "log" in info:
        log = info["log"]
        print(f"\n  Reward breakdown:")
        for key, value in log.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.3f}")
            elif isinstance(value, torch.Tensor):
                print(f"    {key}: {value.item():.3f}")


def test_reset(env):
    """Test environment reset."""
    print(f"\n{'='*70}")
    print("TEST: Environment Reset")
    print(f"{'='*70}")
    
    try:
        obs, info = env.reset()
        print("✅ Reset successful")
        
        if check_observations(obs):
            print("✅ Observations are valid")
        else:
            print("❌ Observations have issues")
            return False
        
        print(f"✅ Observation shape: {obs['policy'].shape}")
        return True
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step(env):
    """Test environment step with random actions."""
    print(f"\n{'='*70}")
    print("TEST: Environment Step")
    print(f"{'='*70}")
    
    try:
        # Random actions (use unwrapped for device)
        device = env.unwrapped.device
        actions = 2.0 * torch.rand(env.action_space.shape, device=device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        print("✅ Step successful")
        
        if check_observations(obs):
            print("✅ Observations are valid")
        else:
            print("❌ Observations have issues")
            return False
        
        print(f"✅ Rewards shape: {rewards.shape}")
        print(f"✅ Mean reward: {rewards.mean():.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_episode(env, max_steps=500, verbose=True):
    """Run a full episode with random actions."""
    print(f"\n{'='*70}")
    print(f"RUNNING EPISODE ({max_steps} steps)")
    print(f"{'='*70}")
    
    obs, info = env.reset()
    
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    
    total_reward = torch.zeros(num_envs, device=device)
    num_successes = 0
    
    for step in range(max_steps):
        # Random actions
        actions = 2.0 * torch.rand(env.action_space.shape, device=device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        total_reward += rewards
        num_successes += terminated.sum().item()
        
        # Print statistics every 100 steps
        if verbose and (step + 1) % 100 == 0:
            print_statistics(step + 1, obs, rewards, info)
        
        # Check for issues
        if not check_observations(obs):
            print(f"❌ Observation issues at step {step}")
            return False
    
    print(f"\n{'='*70}")
    print("EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"Total steps: {max_steps}")
    print(f"Avg reward per env: {total_reward.mean():.3f}")
    print(f"Total successes: {num_successes}")
    print(f"Success rate: {num_successes / (num_envs * max_steps) * 100:.2f}%")
    print(f"{'='*70}\n")
    
    return True


def main():
    """Main test function."""
    print(f"\n{'='*70}")
    print("CITRUS CUTTING ENVIRONMENT TEST")
    print(f"{'='*70}")
    
    # Create configuration directly
    env_cfg = CitrusCuttingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # Create environment
    print("\nCreating environment...")
    try:
        env = gym.make("Isaac-Citrus-Cutting-Direct-v0", cfg=env_cfg)
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Print environment info
    print_env_info(env)
    
    # Run tests
    all_passed = True
    
    # Test 1: Reset
    if not test_reset(env):
        all_passed = False
    
    # Test 2: Step
    if not test_step(env):
        all_passed = False
    
    # Test 3: Full episode
    if all_passed:
        verbose = args_cli.num_envs <= 4  # Only verbose for small num_envs
        if not run_episode(env, max_steps=args_cli.num_steps, verbose=verbose):
            all_passed = False
    
    # Final summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nEnvironment is ready for training!")
        print("\nNext steps:")
        print("  1. python scripts/train_citrus.py --task Isaac-Citrus-Cutting-Direct-v0 --num_envs 64")
        print("  2. tensorboard --logdir logs/rsl_rl")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before training.")
    print(f"{'='*70}\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()