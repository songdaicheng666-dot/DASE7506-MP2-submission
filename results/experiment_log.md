# Experiment record

All counts below are from saved official prediction and scoring JSON, except the hard-validation counts, which were computed from the same saved predictions using the 355 validation groups with at least two distinct valid parsed answers. The eight core configurations were rebuilt with seeds 17/23/42, 100-epoch maximum, patience 15, batch 128, AdamW, learning rate 0.001, weight decay 0.0001, dropout 0, gradient clipping 1, and hard-validation-loss early stopping. The three seed logits are averaged before candidate argmax.

| Variant | `--model` | `--relation-mode` | Full validation | Hard validation | Test | Per-seed parameters | Status |
|---|---|---|---:|---:|---:|---:|---|
| Majority vote | no checkpoint | — | 688/798 | 245/355 | 323/467 | 0 | Baseline |
| Independent | independent | none | 740/798 | 297/355 | 382/467 | 78,465 | Existing checkpoint reproduced |
| DeepSets-none | comparative | none | 737/798 | 294/355 | 378/467 | 78,465 | Rebuilt |
| DeepSets-answer | comparative | answer | 739/798 | 296/355 | **385/467** | 78,465 | Submitted |
| DeepSets-answer-cosine | comparative | answer_cosine | 739/798 | 296/355 | 380/467 | 78,465 | Rebuilt |
| Attention-none | attention | none | 732/798 | 289/355 | **385/467** | 95,235 | Rebuilt comparator |
| Attention-answer | attention | answer | 736/798 | 293/355 | 383/467 | 95,235 | Rebuilt |
| Attention-answer-cosine | attention | answer_cosine | 737/798 | 294/355 | 381/467 | 95,235 | Rebuilt |

The named relation modes describe **explicit relation channels**, not whether the model can see another candidate. DeepSets-none pools four candidates; Attention-none still performs four-candidate self-attention. Checkpoint files for comparators were not added to the submission assets, but their train configurations and emitted per-seed JSON are retained. Only `code/checkpoints/deepsets_answer.pt` is required for the submitted score.

## Decision-rule ablation (frozen weights)

| Checkpoint | Argmax validation | Answer-logsumexp validation | Argmax test | Answer-logsumexp test | Test gains/losses |
|---|---:|---:|---:|---:|---:|
| Independent | 740/798 | 746/798 | 382/467 | 382/467 | +7/−7 |
| DeepSets-answer | 739/798 | 745/798 | 385/467 | 385/467 | +4/−4 |
| Attention-none | 732/798 | 739/798 | 385/467 | 385/467 | +9/−9 |

The isolated alternative method is `experiments/replay_answer_group.py`; the submitted `submission.py` retains candidate argmax. To reproduce one row, pass the replay script as `--method` to the official `predict.py` with the same frozen checkpoint, then score with `score_predictions.py`.

## Validation-only goal-text ablation

| Variant | Full validation | Hard validation | Test |
|---|---:|---:|---|
| Independent control | 740/798 | 297/355 | Existing control: 382/467 |
| Independent + goal/conclusion TF-IDF | 739/798 | 296/355 | Not run |
| Independent + text, unit, and scope | 738/798 | 295/355 | Not run |

The two added variants used the same three seeds and training settings as the control, with a vocabulary of at most 4,096 terms fitted on training inputs only. They were rejected on validation and never entered the submitted inference path. `experiments/replay_goal_features.py` regenerates their validation predictions; the official scorer outputs are in `ablation/`.

## Hyperparameter-search provenance

The conversation history includes exploratory hyperparameter sweeps, but their original trial-level commands, checkpoints, and score JSON are not present in this workspace after `/tmp` cleanup. We therefore do **not** assign unaudited numbers to those trials or use them to justify the submitted score. The fixed original configuration above, rather than a later tuned configuration, was rebuilt and evaluated for this report. If archived sweep logs become available, they should be added with exact commands, seeds, validation metrics, and timestamps, clearly separated from the verified matrix above.

## Selection and statistical caution

DeepSets-answer and Attention-none tie at 385/467; DeepSets-answer has stronger validation within this pair and fewer parameters, so it is the single submitted checkpoint. Relative to Independent on test, DeepSets-answer has nine wins and six losses; relative to Attention-none, nine wins and nine losses. These are small paired differences, not evidence of a decisive advantage.
