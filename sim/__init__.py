"""
Simulation Module - MuJoCo Environment and Control

This module contains:
- PickPlaceEnv: MuJoCo environment for pick-and-place task
- ScriptedExpert: Scripted expert policy for demonstrations
- Utilities for demo collection, rollouts, and visualization

The environment uses a UR5e robot arm with Jacobian-based IK control.
"""

from sim.make_scene import PickPlaceEnv
from sim.expert import ScriptedExpert

__all__ = ['PickPlaceEnv', 'ScriptedExpert']
