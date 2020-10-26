# Ergodic Exploration for Active SLAM

This repository contains implementation of an active SLAM algorithm for the paper "Ergodic Coverage for Exploration-Exploitation in Active SLAM".

## Requirements

Dependencies include:
 - NumPy
 - Matplotlib
 - Autograd
 - Gym
 - tqdm

## Quick Start

To quickly generate the simulation evaluation table in the paper:
~~~
cd eval1
python3 eval2_mc_corner4.py # or: eval2_mc_corner.py eval2_mc_uniform.py
~~~

To quickly plot the trajectories generated from MPC approach and our ergodic exploration approach:
~~~
cd active_ekf_demo
python3 fmi_demo_8.py
~~~
