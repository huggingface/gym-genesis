from setuptools import setup

setup(
    name="gym_genesis",
    version="0.1.0",
    description="A gym environment for GENESIS",
    author="Jade Choghari",
    author_email="jchoghar@uwaterloo.ca",
    install_requires=[
        "gymnasium",
    ],
    extras_require={
        "lerobot": ["lerobot @ git+https://github.com/huggingface/lerobot.git@main"],
    },
    packages=["gym_genesis"],  # adjust if needed
    include_package_data=True,
)
