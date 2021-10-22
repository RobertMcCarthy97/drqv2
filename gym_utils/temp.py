from dm_control import manipulation
# ‘ALL‘ is a tuple containing the names of all of the environments.
print('\n'.join(manipulation.ALL))

print('\n'.join(manipulation.get_environments_by_tag('vision')))

env = manipulation.load('stack_3_bricks_vision', seed=42)

obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()