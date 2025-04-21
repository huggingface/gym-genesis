from gymnasium.envs.registration import register

register(
    id="gym_genesis/CubePick-v0",
    entry_point="gym_genesis.env:GenesisEnv",
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={"task": "cube", "enable_pixels": False},
)
