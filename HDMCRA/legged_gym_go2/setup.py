from setuptools import find_packages
from distutils.core import setup

setup(name='hdmcr-unitree-rl-gym',
      version='2.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots (HDMCRA fork)',
      install_requires=['isaacgym', 'hdmcr-rsl-rl', 'matplotlib', 'numpy>=1.20', 'tensorboard', 'pyyaml'])
