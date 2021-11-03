# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg

from gym_utils import gym_wrappers
# import numpy as np

def create_bullet_env(task='reach'):
    camera_setup = [
        {
            'cameraEyePosition': [-1.0, 0.25, 0.6],
            'cameraTargetPosition': [-0.6, 0.05, 0.2],
            'cameraUpVector': [0, 0, 1],
            'render_width': 84,
            'render_height': 84
        }
    ]
    
    env = pmg.make_env(
        # task args ['reach', 'push', 'slide', 'pick_and_place', 
        #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
        task=task,
        gripper='parallel_jaw',
        num_block=4,  # only meaningful for multi-block tasks
        render=False,
        binary_reward=False,
        max_episode_steps=50,
        # image observation args
        image_observation=True,
        depth_image=False,
        goal_image=False,
        visualize_target=True,
        camera_setup=camera_setup,
        observation_cam_id=0,
        goal_cam_id=0,
        # curriculum args
        use_curriculum=False,
        num_goals_to_generate=90)
    
    # env = gym_wrappers.ObservationWrapper(env)
    
    env = gym_wrappers.DMEnvFromGym(env)
    
    return env