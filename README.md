# DASE7506 MP2 — Candidate Selection Submission

This repository contains the complete course bank and official evaluation tools, plus our completed candidate selector and research evidence. The frozen submitted method is a three-seed **DeepSets-answer** ensemble. Its official test result is **385/467 = 82.44111349036403%**.

- [English report (PDF)](report/report.pdf) and [editable report source](report/report.html)
- [Exact installation, training, and evaluation commands](code/README.md)
- [Submitted checkpoint](code/checkpoints/deepsets_answer.pt) and [training summary](code/checkpoints/deepsets_answer.json)
- [Official result JSON and complete experiment record](results/README.md)
- [Hashes and local resource measurements](results/SHA256.md)

The checkpoint is under 16 MiB. Prediction needs no network access and does not open reference answers. The starter package's `code/PACKAGE_MANIFEST.json` describes the original course release, not this modified submission; use the hashes in `results/SHA256.md` to identify this version.

To reproduce the reported score, clone this fork, enter `code/`, install `requirements.txt` in Python 3.12, run `check_data.py`, and follow the frozen-checkpoint commands in `code/README.md`. The official scorer's `accuracy` field is the number to submit.
