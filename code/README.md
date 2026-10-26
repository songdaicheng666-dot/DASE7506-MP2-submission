# MP2 — Code and Running Instructions

Start with the [Project Guide](../guide/GUIDE.md) for the task description, example, research questions, assessment, deadline and submission rules.

This fork contains a completed candidate selector in `submission.py`, its training program in `train_scorers.py`, and the frozen submitted asset at `checkpoints/deepsets_answer.pt`. The submitted model is the three-seed **DeepSets-answer** ensemble. It selects 385 of 467 test questions correctly (82.441113%). The eight-method comparison and discarded ablations are documented in [`../report/report.pdf`](../report/report.pdf) and [`../results/`](../results/). Run the commands below from **`code/`**.

## 1. Install and check the data

Clone this complete public repository and use **Python 3.12**:

```bash
git clone https://github.com/songdaicheng666-dot/DASE7506-MP2-submission.git
cd DASE7506-MP2-submission/code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

Alternatively, download this fork as a ZIP, extract it, and enter its `code/` directory before running the environment/install commands above.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The submitted implementation needs NumPy and PyTorch at the versions pinned in `requirements.txt`; prediction and scoring run offline.

## 2. Submitted method and exact evaluation

`submission.predict(bank, arrays, checkpoint)` is implemented. With the submitted checkpoint, it loads train-fitted feature statistics and three DeepSets-answer state dictionaries. Without a checkpoint it uses the training-free majority baseline.

- `bank` is a list in fixed question order. Each row has `id`, `question` and four `candidates`; each candidate has `index` and `text`.
- `arrays['embeddings']` is a float32 NumPy array of shape **[N, 4, 1024]**, aligned with the candidate order. Candidate text and these fixed representations are available to every method.
- `checkpoint` is a `pathlib.Path` or `None`. Save every learned weight, transformation, statistic and configuration needed for inference; return exactly one **Python integer from 0 to 3** per question, in bank order.
- Training references are used only by `train_scorers.py`; `predict()` uses no references or question IDs.

Evaluate validation with the frozen submitted asset:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/usr/bin/time -p python predict.py --method submission.py \
  --checkpoint checkpoints/deepsets_answer.pt --split validation \
  --output /tmp/mp2-validation-predictions.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/usr/bin/time -p python score_predictions.py --split validation \
  --predictions /tmp/mp2-validation-predictions.json \
  --output /tmp/mp2-validation-score.json
```

Expected official validation count: **739/798**, with **296/355** on the hard subset. Omit `--checkpoint` only for the separate majority baseline. The prediction runner never opens reference answers.

Evaluate test without changing the method or checkpoint:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/usr/bin/time -p python predict.py --method submission.py \
  --checkpoint checkpoints/deepsets_answer.pt --split test \
  --output /tmp/mp2-test-predictions.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/usr/bin/time -p python score_predictions.py --split test \
  --predictions /tmp/mp2-test-predictions.json \
  --output /tmp/mp2-test-score.json
```

Expected official test result: **385/467, 82.44111349036403%**. Submit the scorer's `accuracy` field, not a rounded value or a value embedded in a prediction file. The scorer grades parsed final numeric answers, not reasoning validity.

## 3. Rebuild the training and ablation evidence

The submitted checkpoint can be used without retraining. To rebuild it independently, use the same fixed configuration and train-only fit of local-feature statistics:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train_scorers.py --model comparative --relation-mode answer \
  --seeds 17 23 42 --epochs 100 --patience 15 --batch-size 128 \
  --learning-rate 0.001 --weight-decay 0.0001 --dropout 0 \
  --selection-metric hard_loss --output /tmp/mp2-retrained-deepsets-answer.pt
```

For the other neural rows in the main table, keep every flag above unchanged and vary only `--model`/`--relation-mode`: `independent/none`, `comparative/none`, `comparative/answer_cosine`, `attention/none`, `attention/answer`, and `attention/answer_cosine`. Use a distinct `/tmp` output for each run. The majority row needs no training or checkpoint. The answer-group decision replay is `../experiments/replay_answer_group.py` as an alternative `--method` with the relevant original checkpoint. The validation-only goal-text replays use `../experiments/replay_goal_features.py --mode text` or `--mode text_unit` with `--output-prefix` pointing to a temporary directory. Neither replay is imported by the submitted `submission.py`.

The unedited official JSON evidence is in `../results/core/` and `../results/ablation/`; per-seed training summaries are in `checkpoints/deepsets_answer.json` and `../results/training/`. See `../results/README.md` for organization and `../results/SHA256.md` for hashes and local resource measurements.

## 4. Data and files

| Files | Purpose |
|---|---|
| `submission.py` | Implemented selector; no reference lookup at inference. |
| `train_scorers.py` | Reproducible three-seed neural training. |
| `checkpoints/deepsets_answer.pt` | Sole submitted inference asset. |
| `predict.py` | Runs your frozen method; writes predictions without reading references. |
| `score_predictions.py`, `common.py` | Fixed independent scorer, numeric parser and data loading. |
| `check_data.py` | Checks hashes, split disjointness and feature/reference alignment. |
| `data/candidates/gsm8k_{split}.jsonl` | Questions and four candidate traces. |
| `data/candidates/gsm8k_{split}.npz` | Fixed whole-trace embeddings; no model weights or correctness labels. |
| `data/gsm8k_{split}.jsonl` | Aligned references: training supervision, validation selection, test scoring only. |
| `data/manifest.json` | Protocol, counts, provenance and data hashes. |
| `data/LICENSE.txt`, `data/NOTICE.txt` | Source license and attribution. |
| `PACKAGE_MANIFEST.json` | Original course-release hashes; changed submission files have new hashes in `../results/SHA256.md`. |

Protocol: **`dase7506-mp2-open-hard-verifier-v5`**. Splits contain **4,755 / 798 / 467** train/validation/test questions and do not overlap. Every group has a correct final answer. Training and validation include easier groups; test questions have at least two different valid parsed answers. Candidate order is fixed. Test references are public for self-reporting and peer reproduction; they must not enter training, lookup tables or method selection.

The instructor may supply a separate audit bank without references:

```bash
python predict.py --method submission.py --checkpoint checkpoints/deepsets_answer.pt \
  --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

The same frozen method must accept that bank without retraining or hard-coded sample counts. It should use the bank/arrays passed to `predict()`, rather than opening reference files or assuming that every input is the public test set.

## 5. Measurement and reproduction

The [Guide](../guide/GUIDE.md#3-constraints) defines the resource limits and measurement rules: **30 seconds** for complete CPU prediction plus scoring with **four threads**, and **16 MiB** of additional inference assets. The runner's JSON timing fields do not include all process startup; measure externally and record your hardware.

The selected checkpoint has 78,465 parameters per seed and is 952,031 bytes. On a local Intel Core i7-12700H, one fresh test prediction process took 1.27 seconds and the independent scoring process took 0.11 seconds with the four-thread settings above. The course reference CPU determines formal eligibility. A peer can run the frozen checkpoint without retraining. See the [submission and peer-review rules](../guide/GUIDE.md#4-submission-and-peer-review).

## 6. Attribution and disclosure

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), distributed under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the model revision is recorded in the data manifest. No Qwen weights were added. Keep `data/LICENSE.txt` and `data/NOTICE.txt` when redistributing this package. Substantive Codex assistance was used for implementation, experiments, analysis, and report drafting; see the report for the full disclosure.
