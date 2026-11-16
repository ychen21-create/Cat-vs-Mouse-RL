# Cat vs. Mouse — Reinforcement Learning Comparison

**A comparative study of value-based and policy-based reinforcement learning in a custom GridWorld environment.**

> **Course project code + original experimental artifacts.** The final presentation slides and written report are maintained separately and are not part of this repository.

## Project overview

A cat learns to capture a mouse on a grid with obstacles, noisy actions and either a static or a moving target. We compare three agents:

- **Q-learning** — tabular, off-policy value-based RL;
- **SARSA** — tabular, on-policy value-based RL;
- **REINFORCE** — neural-network, policy-gradient RL.

The project includes a Tkinter-based visual dashboard, a Gymnasium-compatible optional wrapper, training and evaluation scripts, and saved artifacts from **108 main training runs** plus **48 hyperparameter trials**.

### Environment setups

| Dimension | Settings |
|---|---|
| Map layout | `empty` (5×5), `obstacles` (5×5), `hard` (7×7) |
| Mouse behavior | `static`, `moving` (random legal move or stay) |
| Reward | `standard`, `shaped` |
| Actions | up / down / left / right (discrete) |
| Observation | cat + mouse positions (tabular state encoding; 4 normalized floats for REINFORCE) |
| Action slip probability | 0.10 in the released experiments |
| Episode cap | 65 steps (5×5) or 100 steps (`hard`) |

Both mouse modes are **variants of the same custom GridWorld framework**, not necessarily two independently named Gymnasium benchmark environments. The Gymnasium adapter is optional: the experiments use the custom environment directly.

## Quick start on Windows / VS Code

1. Install Python 3.10+ (the supplied runs used Python 3.13 on Windows).
2. Open this repository's folder in VS Code, open a terminal **in the repository root**, and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Launch the visual comparison:

   ```powershell
   python visual_game.py
   ```

The repository includes the **seed 0 pre-trained model for every configuration**, plus seeds 1 and 2. Select any map, mouse mode and reward type from the UI. Click **Play**, **Pause**, **Step** or **Reset**. If a model load fails, verify you are running the code from the repository root and the model file is present under `results/`.

**Terminal demonstration** (also uses the supplied models):

```powershell
python demo.py --map hard --mode moving --reward standard --algorithm q_learning
```

## Reproducing the experiments

This repository ships with its original experimental outputs, so **you do not need to rerun training to use the GUI or inspect the CSV results**.

The recorded main runs use **1000 training episodes**, **100 evaluation episodes per trained model**, **3 independent training seeds (0, 1, 2)**, three maps, two mouse modes, two reward functions and three algorithms (108 runs in total).

Default parameters: Q-learning / SARSA learning rate α=0.2, γ=0.97, ε-decay=0.996; REINFORCE Adam learning rate=0.003, hidden width=64, entropy coefficient=0.01, γ=0.97. The action-slip probability is 0.10.

To reproduce the complete main suite in a **separate output folder**, without overwriting the original published artifacts:

```powershell
python run_experiments.py --episodes 1000 --eval-episodes 100 --seeds 3 --out results/reproduced_main
```

For a **quick test** (two selected algorithms; short and lightweight):

```powershell
python run_experiments.py --episodes 30 --eval-episodes 10 --seeds 1 --maps empty --modes static --rewards standard --algorithms q_learning sarsa --out results/smoke_test
```

Hyperparameter study: 3 algorithms × 8 configurations × 2 seeds = **48 trials**; map=`obstacles`, mouse mode=`moving`, reward=`standard`, 350 training episodes and 50 evaluation episodes per trial.

```powershell
python parameter_study.py --episodes 350 --seeds 2 --map obstacles --mode moving --out results/tuning_reproduced
```

**Note:** Both training scripts write to their selected output folders. Using the default output locations can overwrite original result CSVs.

## Included experimental data

| Path | Meaning |
|---|---|
| `results/evaluation_summary.csv` | One row per main trained model (108 rows) |
| `results/aggregate_summary.csv` | Means / standard deviations across three training seeds (36 configurations) |
| `results/*_training.csv` | Per-episode training histories (108 files) |
| `results/*_seed*.npy` | 72 tabular Q-table models |
| `results/*_seed*.pt` | 36 REINFORCE PyTorch state dictionaries |
| `results/tuning/hyperparameter_trials.csv` | The 48 actual hyperparameter trials |
| `results/learning_curves.png` | Learning curve figure from the original main runs |
| `results/run_metadata.json` | Recorded Python, OS, CPU, Torch version and run parameters |

### Hyperparameter study — selected candidates

Candidates below are **selected by mean success rate, then mean reward** among two training seeds each. This is not evidence of a globally optimal configuration.

| Algorithm | Selected parameters | Mean success (2 seeds) | Mean reward |
|---|---|---:|---:|
| Q-learning | α=0.3, γ=0.99, ε-decay=0.997 | 100% | 8.479 |
| SARSA | α=0.1, γ=0.90, ε-decay=0.997 | 100% | 8.306 |
| REINFORCE | LR=0.005, hidden=32, entropy=0.0 | 92% | -13.698 |

These tuning results are specific to the `obstacles/moving/standard` setup, with shorter training than the main 1000-episode experiments. Do not mix tuned and main results without clearly identifying their differing conditions.

### Interpretation and limitations

- Learning curves, final success rate, cumulative reward, time and episode length should be considered together.
- REINFORCE can show substantial sensitivity to seed, environment difficulty and reward design; a two-seed tuning result is only preliminary.
- `standard` and `shaped` rewards use **different terminal rewards and step penalties**, so their raw returns are not directly comparable as if only distance shaping had changed.
- Gymnasium is an **optional integration**, and static/moving targets are two tasks within the same custom environment family.
- All models and CSV results here are from the uploaded project; no results were recreated for repository documentation.

## Project files

```text
agents.py                  # Q-learning, SARSA, REINFORCE and evaluation
cat_mouse_env.py           # GridWorld tasks + optional Gymnasium adapter
run_experiments.py         # Main experiment suite and aggregate summaries
parameter_study.py         # Hyperparameter grid search
demo.py                    # Console demonstration
visual_game.py             # Interactive Tkinter dashboard
requirements.txt           # Required Python packages
tests/test_project.py      # Basic automated tests
results/                   # Experimental outputs + pre-trained models
```

## Run automated tests

```powershell
python -m pip install pytest
python -m pytest -q
```

The project includes no third-party dataset and does not require a remote API key. Model files are supplied for classroom reproducibility; only load PyTorch checkpoints you trust.

## Publishing / coursework

The GitHub repository is for **source code, setup, model checkpoints and experiment artifacts**. The final PowerPoint and written report are submitted separately, as required by the course. Confirm with the instructor whether two task variants satisfy the rubric's environment-count requirement.
