"""Candidate-selection methods for DASE7506 MP2.

Without a checkpoint, ``predict`` uses the training-free majority-vote
baseline. Learned checkpoints support self-only, DeepSets, or permutation-
equivariant four-candidate attention context.
"""

from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from common import parse_answer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
LOCAL_FEATURE_NAMES = (
    "parseable",
    "answer_sign",
    "answer_log_magnitude",
    "is_integer",
    "log_text_length",
    "log_number_count",
    "question_number_coverage",
)
RELATION_FEATURE_NAMES = (
    "answer_vote_fraction",
    "mean_embedding_cosine",
    "max_embedding_cosine",
)
PAIRWISE_RELATION_NAMES = (
    "answer_equal",
    "embedding_cosine",
)
FEATURE_SCHEMA_VERSION = 1
EMBEDDING_DIM = 1024
HIDDEN_DIM = 64


def _select_by_answer_vote(candidates):
    """Select the earliest candidate whose parsed answer has most votes."""
    answers = [parse_answer(candidate["text"]) for candidate in candidates]
    votes = Counter(answer for answer in answers if answer is not None)
    if not votes:
        return 0

    most_votes = max(votes.values())
    return next(
        index
        for index, answer in enumerate(answers)
        if answer is not None and votes[answer] == most_votes
    )


def _numeric_tokens(text):
    """Return normalized numeric tokens found by the benchmark-style regex."""
    tokens = []
    for match in NUMBER_RE.finditer(str(text)):
        try:
            tokens.append(str(Fraction(match.group().replace(",", ""))))
        except (ValueError, ZeroDivisionError):
            continue
    return tokens


def _answer_features(text):
    parsed = parse_answer(text)
    if parsed is None:
        return 0.0, 0.0, 0.0, 0.0

    value = Fraction(parsed)
    sign = float((value > 0) - (value < 0))
    try:
        log_magnitude = math.log1p(abs(float(value)))
    except OverflowError:
        log_magnitude = 20.0
    log_magnitude = min(log_magnitude, 20.0)
    return 1.0, sign, log_magnitude, float(value.denominator == 1)


def extract_feature_parts(bank, embeddings):
    """Build L2-normalized embeddings and unstandardized local features."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, EMBEDDING_DIM):
        raise ValueError("Expected embeddings with shape [N, 4, 1024]")
    if len(bank) != embeddings.shape[0]:
        raise ValueError("Bank and embeddings have different lengths")

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, np.float32(1e-12))
    local = np.empty((len(bank), 4, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)

    for question_index, row in enumerate(bank):
        question_numbers = set(_numeric_tokens(row["question"]))
        denominator = max(len(question_numbers), 1)
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            numeric_tokens = _numeric_tokens(text)
            candidate_numbers = set(numeric_tokens)
            coverage = len(question_numbers.intersection(candidate_numbers)) / denominator
            local[question_index, candidate_index] = (
                *_answer_features(text),
                math.log1p(len(text)),
                math.log1p(len(numeric_tokens)),
                coverage,
            )

    return normalized_embeddings.astype(np.float32, copy=False), local


def combine_features(normalized_embeddings, local, local_mean, local_std):
    """Standardize local features and concatenate them to embeddings."""
    local_mean = np.asarray(local_mean, dtype=np.float32).reshape(1, 1, -1)
    local_std = np.asarray(local_std, dtype=np.float32).reshape(1, 1, -1)
    standardized = (local - local_mean) / local_std
    return np.concatenate((normalized_embeddings, standardized), axis=-1).astype(
        np.float32, copy=False
    )


def extract_relation_features(bank, normalized_embeddings):
    """Build exact answer-consistency and embedding-similarity signals."""
    relations = np.empty(
        (len(bank), 4, len(RELATION_FEATURE_NAMES)), dtype=np.float32
    )
    cosine = np.einsum(
        "nid,njd->nij",
        normalized_embeddings,
        normalized_embeddings,
        optimize=True,
    )
    diagonal = np.arange(4)
    mean_cosine = (cosine.sum(axis=-1) - cosine[:, diagonal, diagonal]) / 3.0
    cosine_without_self = cosine.copy()
    cosine_without_self[:, diagonal, diagonal] = -np.inf
    max_cosine = cosine_without_self.max(axis=-1)

    for question_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        for candidate_index, answer in enumerate(answers):
            vote_fraction = 0.0 if answer is None else counts[answer] / 4.0
            relations[question_index, candidate_index] = (
                vote_fraction,
                mean_cosine[question_index, candidate_index],
                max_cosine[question_index, candidate_index],
            )
    return relations


def prepare_relation_features(relations, relation_mean, relation_std, relation_mode):
    """Standardize and enable only the pre-registered relation channels."""
    if relation_mode not in {"none", "answer", "answer_cosine"}:
        raise ValueError("Unknown relation mode")
    standardized = (
        relations - np.asarray(relation_mean, dtype=np.float32).reshape(1, 1, -1)
    ) / np.asarray(relation_std, dtype=np.float32).reshape(1, 1, -1)
    enabled = {
        "none": (0.0, 0.0, 0.0),
        "answer": (1.0, 0.0, 0.0),
        "answer_cosine": (1.0, 1.0, 1.0),
    }[relation_mode]
    return (standardized * np.asarray(enabled, dtype=np.float32)).astype(
        np.float32, copy=False
    )


def extract_pairwise_relations(bank, normalized_embeddings):
    """Build full answer-equality and embedding-cosine relation matrices."""
    if len(bank) != len(normalized_embeddings):
        raise ValueError("Bank and embeddings have different lengths")
    cosine = np.einsum(
        "nid,njd->nij",
        normalized_embeddings,
        normalized_embeddings,
        optimize=True,
    ).astype(np.float32, copy=False)
    pairwise = np.zeros(
        (len(bank), 4, 4, len(PAIRWISE_RELATION_NAMES)), dtype=np.float32
    )
    pairwise[..., 1] = cosine

    for question_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        for left, left_answer in enumerate(answers):
            if left_answer is None:
                continue
            for right, right_answer in enumerate(answers):
                if left != right and right_answer is not None:
                    pairwise[question_index, left, right, 0] = float(
                        left_answer == right_answer
                    )

    diagonal = np.arange(4)
    pairwise[:, diagonal, diagonal, :] = 0.0
    return pairwise


def prepare_pairwise_relations(relations, relation_mode):
    """Enable the registered pairwise channels without changing their scale."""
    if relation_mode not in {"none", "answer", "answer_cosine"}:
        raise ValueError("Unknown relation mode")
    relations = np.asarray(relations, dtype=np.float32)
    if relations.ndim != 4 or relations.shape[1:] != (4, 4, 2):
        raise ValueError("Expected pairwise relations with shape [N, 4, 4, 2]")
    enabled = {
        "none": (0.0, 0.0),
        "answer": (1.0, 0.0),
        "answer_cosine": (1.0, 1.0),
    }[relation_mode]
    return (relations * np.asarray(enabled, dtype=np.float32)).astype(
        np.float32, copy=False
    )


class SetScorer(nn.Module):
    """Candidate scorer with self-only, DeepSets, or attention context."""

    def __init__(self, input_dim, context_mode, dropout=0.0):
        super().__init__()
        if context_mode not in {"independent", "comparative", "attention"}:
            raise ValueError(
                "context_mode must be independent, comparative, or attention"
            )
        self.context_mode = context_mode
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        if context_mode == "attention":
            self.query = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.key = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.value = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.attention_output = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.attention_norm = nn.LayerNorm(HIDDEN_DIM)
            self.answer_scale = nn.Parameter(torch.zeros(()))
            self.cosine_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features, relation_features=None):
        hidden = self.dropout(self.encoder(features))
        if self.context_mode == "attention":
            query = self.query(hidden)
            key = self.key(hidden)
            value = self.value(hidden)
            attention_logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
                HIDDEN_DIM
            )
            if relation_features is not None:
                expected_shape = (*hidden.shape[:-1], hidden.shape[1], 2)
                if relation_features.shape != expected_shape:
                    raise ValueError(
                        "Expected attention relations with shape [N, 4, 4, 2]"
                    )
                attention_logits = (
                    attention_logits
                    + self.answer_scale * relation_features[..., 0]
                    + self.cosine_scale * relation_features[..., 1]
                )
            attention_weights = torch.softmax(attention_logits, dim=-1)
            attended = torch.matmul(attention_weights, value)
            context = self.attention_norm(
                hidden + self.attention_output(attended)
            )
            combined = torch.cat(
                (hidden, context, hidden - context, hidden * context), dim=-1
            )
            return self.head(self.dropout(combined)).squeeze(-1)

        if self.context_mode == "independent":
            context_mean = hidden
            context_max = hidden
        else:
            context_mean = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            context_max = hidden.max(dim=1, keepdim=True).values.expand_as(hidden)
        relative = hidden - context_mean
        if relation_features is not None:
            if relation_features.shape[:-1] != hidden.shape[:-1]:
                raise ValueError("Relation features are not aligned with candidates")
            if relation_features.shape[-1] > relative.shape[-1]:
                raise ValueError("Too many relation feature channels")
            relative = relative.clone()
            relative[..., : relation_features.shape[-1]] += relation_features
        combined = torch.cat((hidden, context_mean, context_max, relative), dim=-1)
        return self.head(self.dropout(combined)).squeeze(-1)


def _load_bundle(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint}")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint feature schema")
    if bundle.get("context_mode") not in {
        "independent",
        "comparative",
        "attention",
    }:
        raise ValueError("Checkpoint has an invalid context mode")
    if bundle.get("relation_mode", "none") not in {
        "none",
        "answer",
        "answer_cosine",
    }:
        raise ValueError("Checkpoint has an invalid relation mode")
    if (
        bundle.get("context_mode") == "independent"
        and bundle.get("relation_mode", "none") != "none"
    ):
        raise ValueError("Independent checkpoints cannot use cross-candidate relations")
    if (
        bundle.get("context_mode") == "attention"
        and bundle.get("attention_version") != 1
    ):
        raise ValueError("Checkpoint has an invalid attention version")
    if not bundle.get("state_dicts"):
        raise ValueError("Checkpoint contains no model weights")
    if not 0.0 <= float(bundle.get("dropout", 0.0)) < 1.0:
        raise ValueError("Checkpoint has an invalid dropout value")
    return bundle


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    if checkpoint is None:
        return [int(_select_by_answer_vote(row["candidates"])) for row in bank]

    bundle = _load_bundle(checkpoint)
    normalized_embeddings, local = extract_feature_parts(bank, arrays["embeddings"])
    features = combine_features(
        normalized_embeddings,
        local,
        bundle["local_mean"].numpy(),
        bundle["local_std"].numpy(),
    )
    feature_tensor = torch.from_numpy(features)
    relation_mode = bundle.get("relation_mode", "none")
    relation_tensor = None
    if bundle["context_mode"] == "attention" and relation_mode != "none":
        relations = extract_pairwise_relations(bank, normalized_embeddings)
        prepared_relations = prepare_pairwise_relations(relations, relation_mode)
        relation_tensor = torch.from_numpy(prepared_relations)
    elif relation_mode != "none":
        relations = extract_relation_features(bank, normalized_embeddings)
        prepared_relations = prepare_relation_features(
            relations,
            bundle["relation_mean"].numpy(),
            bundle["relation_std"].numpy(),
            relation_mode,
        )
        relation_tensor = torch.from_numpy(prepared_relations)

    logits_sum = torch.zeros((len(bank), 4), dtype=torch.float32)
    with torch.inference_mode():
        for state_dict in bundle["state_dicts"]:
            model = SetScorer(
                features.shape[-1],
                bundle["context_mode"],
                dropout=float(bundle.get("dropout", 0.0)),
            )
            model.load_state_dict(state_dict)
            model.eval()
            logits_sum.add_(model(feature_tensor, relation_tensor))
    choices = (logits_sum / len(bundle["state_dicts"])).argmax(dim=1).tolist()
    return [int(choice) for choice in choices]
