# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt

import gym_wrappers
import numpy as np

camera_setup = [
    {
        'cameraEyePosition': [-1.0, 0.25, 0.6],
        'cameraTargetPosition': [-0.6, 0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    }
]

env = pmg.make_env(
    # task args ['reach', 'push', 'slide', 'pick_and_place', 
    #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
    task='reach',
    gripper='parallel_jaw',
    num_block=4,  # only meaningful for multi-block tasks
    render=False,
    binary_reward=False,
    max_episode_steps=5,
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

env = gym_wrappers.ObservationWrapper(env)

env = gym_wrappers.DMEnvFromGym(env)

random_state = np.random.RandomState(42)
num_sequences = 5
for _ in range(num_sequences):

  timestep = env.reset()
  while True:
    action = random_state.uniform(env.action_spec().minimum, env.action_spec().maximum, env.action_spec().shape)
    timestep = env.step(action)
    print(timestep.observation.shape)
    print(timestep.reward)
    plt.imshow(timestep.observation)
    plt.pause(0.00001)
    if timestep.last():
      break
print('DONE')