from dm_control import suite
# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")
# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
    env = suite.load(domain_name, task_name)


from dm_control.suite.wrappers import pixels

env = suite.load("cartpole", "swingup")
# Replace existing features by pixel observations:
env_only_pixels = pixels.Wrapper(env)
# Pixel observations in addition to existing features.
env_plus_pixels = pixels.Wrapper(env, pixels_only=False)