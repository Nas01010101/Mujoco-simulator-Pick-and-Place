"""
ML Module - Imitation Learning Algorithms

This module contains implementations of various imitation learning algorithms:
- Behavior Cloning (BC)
- DAgger (Dataset Aggregation)
- GAIL (Generative Adversarial Imitation Learning)
- Diffusion Policy

Each algorithm can be trained via the Streamlit UI or command line.
"""

from ml.train_bc import BCPolicy
from ml.dataset import DemoDataset

__all__ = ['BCPolicy', 'DemoDataset']
