from gymnasium.envs.registration import register

register(
    id="gym_genesis/CubePick-v0",
    entry_point="gym_genesis.env:GenesisEnv",
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={
        "task": "cube",
        "enable_pixels": False,
        "num_envs": 10,
        "observation_height": 480,
        "observation_width": 640,
        "env_spacing": (1.0, 1.0),
        "camera_capture_mode": "global",
    },
)

register(
    id="gym_genesis/CubeStack-v0",
    entry_point="gym_genesis.env:GenesisEnv",
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={
        "task": "cube_stack",
        "enable_pixels": False,
        "num_envs": 10,
        "observation_height": 480,
        "observation_width": 640,
        "env_spacing": (1.0, 1.0),
        "camera_capture_mode": "global",
    },
)
