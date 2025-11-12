## RNN Architectures for Sentiment Classification

End-to-end implementation for the MSML641 assignment exploring recurrent neural networks on the IMDB sentiment dataset. The project delivers reproducible preprocessing, configurable training scripts, automated evaluation utilities, and documentation assets.

### 1. Environment Setup
- **Python**: 3.9 (recommended). Any 3.9–3.11 build supported by PyTorch works.
- Create an isolated environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- On first use, `nltk` downloads the `punkt` tokenizer automatically (no manual step needed).

### 2. Dataset Preparation
- Place the provided `IMDB Dataset.csv` under `data/`.
- The preprocessing pipeline enforces a deterministic 25k/25k train–test split and derives a 10% validation slice from the training data.
- Cleaning: lowercase, strip HTML breaks, remove punctuation/special characters, collapse whitespace.
- Tokenisation: `nltk.word_tokenize`
- Vocabulary: top 10,000 most frequent tokens (PAD=0, UNK=1).
- Sequence lengths explored: 25, 50, and 100 tokens (pad/truncate).

### 3. Core Modules
| Script | Purpose |
| --- | --- |
| `src/preprocess.py` | Cleans text, builds vocabulary, tokenises reviews, and caches padded tensors. |
| `src/models.py` | Defines RNN, LSTM, and Bidirectional LSTM classifiers with selectable activations. |
| `src/train.py` | Loads JSON configs, runs experiment(s), logs metrics, and stores history/checkpoints. |
| `src/evaluate.py` | Aggregates metrics, generates plots, and prints top/bottom runs. |
| `src/utils.py` | Shared helpers for seeding, logging, metrics, and file IO. |

### 4. Running Experiments

#### 4.1 Quick Start - Run All Required Experiments
To run the complete experiment suite required by the assignment:
```bash
./run_all_experiments.sh
```
This script executes all 12 experiments from `configs/full_experiments.json` covering:
- Architecture variations (RNN, LSTM, BiLSTM)
- Activation functions (Sigmoid, ReLU, Tanh)
- Optimizers (Adam, SGD, RMSProp)
- Sequence lengths (25, 50, 100)
- Gradient clipping (enabled/disabled)

#### 4.2 Custom Experiments
All experiment parameters live in JSON config files. The `defaults` block holds shared hyperparameters, while each entry in `experiments` overrides specific values.

Run a specific config file:
```bash
python -m src.train --config configs/full_experiments.json
```

For quick testing with just the baseline:
```bash
python -m src.train --config configs/default.json
```

Add `--prefer_gpu` to override the config and force CUDA usage when available.

**Important:** Follow the assignment requirement—change only one factor per experiment entry.

Each run appends results to `results/metrics.csv` and stores histories in `results/runs/<run_id>/`. Logs are printed to console during training.

### 5. Evaluation & Plotting
After recording all runs:
```bash
python -m src.evaluate --hardware "CPU-only"
```

Outputs produced in `results/`:
- `metrics_summary.csv`: top configurations sorted by macro-F1.
- `plots/accuracy_f1_vs_sequence_length.png`: performance vs sequence length.
- `plots/loss_curves_best_vs_worst.png`: training vs validation loss for best and worst runs (by F1).

### 6. Logging & Reproducibility
- Global seeds fixed to 42 for `random`, `numpy`, and PyTorch.
- Hardware info is echoed in evaluation output; update the `--hardware` flag to match your machine.
- Training progress is logged to console with detailed epoch metrics.
- Processed datasets cached in `data/processed/` (reused unless `--force_rebuild` is added in preprocessing calls).

### 7. Report Checklist (`report.pdf`)
Include the following sections:
1. **Dataset Summary** – preprocessing pipeline, vocabulary size, average/median review length (see `data/processed/metadata.json`).
2. **Model Configuration** – embedding dim=100, hidden size=64, 2 layers, dropout 0.3–0.5, batch 32, sigmoid output, BCE loss.
3. **Comparative Analysis** – summary table (mirrors `results/metrics.csv`) plus plots above; discuss trends when varying one factor at a time.
4. **Discussion** – identify the best-performing configuration; explain impacts of sequence length, optimizer choice, and gradient clipping on performance/stability.
5. **Conclusion** – recommend the optimal architecture under CPU constraints with justification; document hardware specs.

### 8. Expected Runtime
- ~35–45 seconds per epoch on modern CPU for LSTM (sequence length 50).
- Full sweep across required settings (≈12 runs) typically completes within 1–2 hours on CPU; faster with GPU (`--prefer_gpu` flag).

### 9. Repository Layout
```
├── data/
│   └── IMDB Dataset.csv
├── configs/
│   ├── default.json
│   └── full_experiments.json
├── results/
│   ├── metrics.csv
│   ├── plots/
│   └── runs/
├── src/
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

### 10. Workflow Summary
1. Activate the environment and install dependencies.
2. Run `python -m src.train`; confirm cache creation in `data/processed/`.
3. Edit `configs/default.json` and rerun to cover the experiment grid (one factor per entry).
4. Call `python -m src.evaluate` to refresh plots and summaries.
5. Draft `report.pdf` using generated artefacts, logs, and metrics.