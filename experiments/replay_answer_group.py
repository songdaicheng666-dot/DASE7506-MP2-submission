"""Reproduce the discarded answer-logsumexp decision ablation.

Use as --method with the official prediction runner. This file is not imported
by the submitted inference method.
"""

from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import parse_answer  # noqa: E402
from submission import (  # noqa: E402
    SetScorer,
    _load_bundle,
    combine_features,
    extract_feature_parts,
    extract_relation_features,
    prepare_relation_features,
)


def predict(bank, arrays, checkpoint):
    bundle = _load_bundle(checkpoint)
    relation_mode = bundle.get("relation_mode", "none")
    if bundle["context_mode"] == "attention" and relation_mode != "none":
        raise ValueError("Only attention-none is supported by this replay")
    embeddings, local = extract_feature_parts(bank, arrays["embeddings"])
    features = combine_features(
        embeddings, local, bundle["local_mean"].numpy(), bundle["local_std"].numpy()
    )
    feature_tensor = torch.from_numpy(features)
    relations = None
    if relation_mode != "none":
        raw = extract_relation_features(bank, embeddings)
        prepared = prepare_relation_features(
            raw, bundle["relation_mean"].numpy(),
            bundle["relation_std"].numpy(), relation_mode,
        )
        relations = torch.from_numpy(prepared)
    with torch.inference_mode():
        logits = torch.zeros((len(bank), 4), dtype=torch.float32)
        for state in bundle["state_dicts"]:
            model = SetScorer(features.shape[-1], bundle["context_mode"])
            model.load_state_dict(state)
            model.eval()
            logits += model(feature_tensor, relations)
        logits /= len(bundle["state_dicts"])
        choices = []
        for row, scores in zip(bank, logits):
            groups = defaultdict(list)
            for index, candidate in enumerate(row["candidates"]):
                answer = parse_answer(candidate["text"])
                if answer is not None:
                    groups[answer].append(index)
            if not groups:
                choices.append(int(scores.argmax().item()))
                continue
            best_group = max(
                groups.values(),
                key=lambda indices: (
                    float(torch.logsumexp(scores[indices], dim=0)), -indices[0]
                ),
            )
            choices.append(int(max(best_group, key=lambda i: (float(scores[i]), -i))))
    return choices
