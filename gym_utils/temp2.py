# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt

camera_setup = [
    {
        'cameraEyePosition': [-1.0, 0.25, 0.6],
        'cameraTargetPosition': [-0.6, 0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
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
    depth_image=True,
    goal_image=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=0,
    goal_cam_id=1,
    # curriculum args
    use_curriculum=True,
    num_goals_to_generate=90)

obs = env.reset()
t = 0
for _ in range(5):
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('obs:')
    print(list(obs.keys()))
    print('obs: {}'.format(obs['observation'].shape))
    print('state: ', obs['state'], '\n',
          'desired_goal: ', obs['desired_goal'], '\n',
          'achieved_goal: ', obs['achieved_goal'], '\n',
          'reward: ', reward, '\n')
    plt.imshow(obs['observation'][:,:,3])
    plt.pause(0.00001)
    input()    
    if done:
        print('\nResetting\n')
        env.reset()
