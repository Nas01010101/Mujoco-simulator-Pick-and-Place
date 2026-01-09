# General Structure of a Robotics ML Project

Welcome! This document outlines the architecture of a typical robotics machine learning project, using this codebase as a case study. We'll extrapolate from the specific files here to general concepts.

## High-Level Architecture

A robotics ML project generally consists of two main worlds interacting with each other:

1.  **The Simulation (The World)**: Where the robot lives, moves, and generates data.
2.  **The Machine Learning (The Brain)**: Where the data is processed, and policies (brains) are trained.

The flow usually looks like this:
`Simulation -> Data Collection -> Dataset -> Training -> Policy -> Evaluation (back in Simulation)`

---

## 1. The Simulation (`sim/`)

This directory handles everything related to the physical world (physics, rendering, robot control).

*   **`make_scene.py` (The Environment)**: This is the "God script". It constructs the arena, loads the robot model (MuJoCo XMLs), sets up lighting, and defines the task (e.g., "pick up the red block"). In a generic project, this would be your Gym environment (`env.step()`, `env.reset()`).
*   **`expert.py` (The Teacher)**: A hard-coded script that knows how to solve the task perfectly using cheat information (like exact object coordinates). We use this to generate **Demonstrations** for the robot to imitate.
*   **`collect_demos.py` (The Data Collector)**: Runs the *Expert* in the *Environment* loop and saves the trajectories (states and actions) to files.

## 2. The Machine Learning (`ml/`)

This directory is standard PyTorch territory. It cares about Tensors, not Physics.

*   **`dataset.py` (The Loader)**: A `torch.utils.data.Dataset` class. Its job is to read the raw demonstration files saved by `collect_demos.py` and hand them to the training loop as nice, batched Tensors.
*   **`dataset.py` (The Model)**: (Often in `dataset.py` or separate `models.py`). Defines the Neural Network architecture (e.g., a Multi-Layer Perceptron) that takes an observation (Positions, Velocities) and outputs an action (Joint Velocities).
*   **`train_bc.py` (The Trainer)**: The optimization loop.
    1.  Get a batch of (State, Action) from the **Dataset**.
    2.  Ask the **Model** to predict an Action given the State.
    3.  Compare the Predicted Action vs. the Expert Action (Loss function, e.g., MSE).
    4.  Update the Model weights (Backpropagation).
*   **`eval.py` (The test)**: Loads a trained model and runs it in the simulation to see if it actually works.

---

## 3. The Tutorial Code (`tutorial/`)

Current research projects can be complex. We have distilled the core "Behavior Cloning" workflow into a single 10-minute exercise.

*   **`tutorial_exercise.py`**: A self-contained script where you will implement the key missing pieces of the pipeline: loading data, the training step, and the inference action.
