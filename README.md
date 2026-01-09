# Robot Pick-and-Place: Imitation Learning

A learning project exploring imitation learning for robot manipulation using MuJoCo simulation.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Project Structure

```
├── app.py              # Streamlit web interface
├── ml/                 # Imitation learning algorithms
│   ├── train_bc.py     # Behavior Cloning
│   ├── dagger.py       # DAgger
│   ├── gail.py         # GAIL
│   └── diffusion_policy.py
├── sim/                # MuJoCo environment
│   ├── make_scene.py   # Environment
│   ├── expert.py       # Scripted expert
│   └── rollout.py      # Video generation
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| Behavior Cloning | Supervised learning from demonstrations |
| DAgger | Interactive expert corrections |
| GAIL | Adversarial reward learning |
| Diffusion Policy | Denoising diffusion for actions |

## Limitations

This is an **experimental learning project**, not production-ready code:

- **Success rates vary significantly** depending on hyperparameters, random seeds, and number of demos.
- The pick-and-place task is simplified (magnetic gripper, fixed bin position).
- Policies may fail to grasp, drop the cube, or miss the bin entirely.
- Run `python ml/eval.py` to measure actual performance on your trained models.

## License

MIT
