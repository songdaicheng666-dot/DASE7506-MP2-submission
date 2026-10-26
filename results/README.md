# Experiment evidence

`core/` contains official prediction and score JSON for the eight main methods on validation and test. `ablation/` contains official scoring evidence for the discarded answer-group decision rule and the validation-only goal-text experiments. `training/` stores per-seed training metrics emitted by `train_scorers.py` for the non-submitted neural variants. The submitted DeepSets-answer training metrics sit beside its checkpoint in `code/checkpoints/`.

Only `code/checkpoints/deepsets_answer.pt` is needed for the submitted prediction. Other checkpoints were generated under `/tmp` and are intentionally not submission assets; their exact reconstruction commands are in `code/README.md`. All test results are reproduced evaluations of frozen configurations, not inputs to training.
