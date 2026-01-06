# Robot Pick-and-Place: Imitation Learning

A complete imitation learning pipeline for robot manipulation, with interactive training UI and 4 algorithms.

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
| Behavior Cloning | Supervised learning |
| DAgger | Interactive expert corrections |
| GAIL | Adversarial reward learning |
| Diffusion Policy | Denoising diffusion |

## License

MIT
