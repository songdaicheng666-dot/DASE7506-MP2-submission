"""Reproduce the two discarded goal-text ablations on validation only.

This is research code, not part of the submitted inference method.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import DATA, PROTOCOL, load_inputs  # noqa: E402
from submission import SetScorer, combine_features, extract_feature_parts  # noqa: E402
from train_scorers import load_labels, train_one_seed, evaluate_ensemble  # noqa: E402


WORD_RE = re.compile(r"[a-z]+")
UNIT_ALIASES = (
    ("money", "cent", ("cent", "cents", "penny", "pennies")),
    ("money", "dollar", ("dollar", "dollars")),
    ("duration", "minute", ("minute", "minutes")),
    ("duration", "hour", ("hour", "hours")),
    ("duration", "day", ("day", "days")),
    ("duration", "week", ("week", "weeks")),
    ("duration", "month", ("month", "months")),
    ("duration", "year", ("year", "years")),
    ("distance", "inch", ("inch", "inches")),
    ("distance", "foot", ("foot", "feet")),
    ("distance", "yard", ("yard", "yards")),
    ("distance", "mile", ("mile", "miles")),
    ("distance", "meter", ("meter", "meters")),
    ("distance", "kilometer", ("kilometer", "kilometers")),
    ("mass", "ounce", ("ounce", "ounces")),
    ("mass", "pound", ("pound", "pounds", "lbs")),
    ("mass", "gram", ("gram", "grams")),
    ("mass", "kilogram", ("kilogram", "kilograms")),
    ("volume", "liter", ("liter", "liters")),
    ("volume", "gallon", ("gallon", "gallons")),
)


def question_goal(question):
    question = str(question).strip()
    questions = re.findall(r"[^?]+\?", question)
    if questions:
        return questions[-1].strip()
    parts = re.split(r"(?<=[.!])\s+|\n+", question)
    return next((part.strip() for part in reversed(parts) if part.strip()), "")


def conclusion(candidate_text):
    text = str(candidate_text)
    reasoning, marker, answer = text.rpartition("####")
    if not marker:
        reasoning, answer = text, ""
    reasoning = re.sub(r"\\(?:text|boxed)\{([^{}]*)\}", r" \1 ", reasoning)
    reasoning = re.sub(r"\\[A-Za-z]+", " ", reasoning)
    pieces = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", reasoning)
        if WORD_RE.search(part.lower())
    ]
    return " ".join(pieces[-2:] + ([answer.strip()] if answer.strip() else []))


def tokens(text):
    return WORD_RE.findall(text.lower())


def fit_vocabulary(train_bank):
    df = Counter()
    count = 0
    for row in train_bank:
        docs = [question_goal(row["question"])]
        docs.extend(conclusion(candidate["text"]) for candidate in row["candidates"])
        for document in docs:
            df.update(set(tokens(document)))
            count += 1
    eligible = ((word, freq) for word, freq in df.items() if 2 <= freq <= 0.8 * count)
    vocabulary = [word for word, _ in sorted(eligible, key=lambda x: (-x[1], x[0]))[:4096]]
    idf = np.array(
        [math.log((1 + count) / (1 + df[word])) + 1 for word in vocabulary],
        dtype=np.float32,
    )
    return vocabulary, idf


def unit_label(words):
    word_set = set(words)
    labels = {
        (family, value)
        for family, value, aliases in UNIT_ALIASES
        if word_set.intersection(aliases)
    }
    return next(iter(labels)) if len(labels) == 1 else None


def quantity_label(words):
    word_set = set(words)
    labels = set()
    period = {"day", "days", "week", "weeks", "month", "months", "year", "years",
              "hour", "hours", "minute", "minutes"}
    if word_set.intersection({"each", "every"}) or any(
        word == "per" and (index + 1 == len(words) or words[index + 1] not in period)
        for index, word in enumerate(words)
    ):
        labels.add("per_item")
    if word_set.intersection({"total", "altogether", "combined", "overall"}):
        labels.add("total")
    if word_set.intersection({"left", "remain", "remaining", "remainder"}):
        labels.add("remaining")
    return next(iter(labels)) if len(labels) == 1 else None


def time_label(words):
    labels = set()
    aliases = {
        "daily": {"daily"}, "weekly": {"weekly"},
        "monthly": {"monthly"}, "yearly": {"yearly", "annually"},
    }
    for label, spelling in aliases.items():
        if set(words).intersection(spelling):
            labels.add(label)
    periods = {
        "day": "daily", "days": "daily", "week": "weekly", "weeks": "weekly",
        "month": "monthly", "months": "monthly", "year": "yearly", "years": "yearly",
    }
    for left, right in zip(words, words[1:]):
        if left in {"per", "each", "every", "a"} and right in periods:
            labels.add(periods[right])
    return next(iter(labels)) if len(labels) == 1 else None


def match_state(expected, actual):
    if expected is None or actual is None:
        return (0.0, 0.0, 1.0)
    if isinstance(expected, tuple) and isinstance(actual, tuple) and expected[0] != actual[0]:
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0) if expected == actual else (0.0, 1.0, 0.0)


def goal_features(bank, mode, vocabulary, idf):
    index = {word: i for i, word in enumerate(vocabulary)}
    features = np.zeros((len(bank), 4, 2 if mode == "text" else 11), dtype=np.float32)

    def weighted(words):
        counts = Counter(index[word] for word in words if word in index)
        return {i: count * float(idf[i]) for i, count in counts.items()}

    for row_index, row in enumerate(bank):
        goal_words = tokens(question_goal(row["question"]))
        goal = weighted(goal_words)
        goal_norm = math.sqrt(sum(value * value for value in goal.values()))
        goal_unique = set(goal)
        goal_weight = sum(float(idf[i]) for i in goal_unique)
        if mode == "text_unit":
            goal_labels = (unit_label(goal_words), quantity_label(goal_words), time_label(goal_words))
        for candidate_index, candidate in enumerate(row["candidates"]):
            candidate_words = tokens(conclusion(candidate["text"]))
            candidate_weights = weighted(candidate_words)
            candidate_norm = math.sqrt(sum(value * value for value in candidate_weights.values()))
            dot = sum(value * candidate_weights.get(i, 0.0) for i, value in goal.items())
            features[row_index, candidate_index, 0] = (
                dot / (goal_norm * candidate_norm) if goal_norm and candidate_norm else 0.0
            )
            features[row_index, candidate_index, 1] = (
                sum(float(idf[i]) for i in goal_unique.intersection(candidate_weights)) / goal_weight
                if goal_weight else 0.0
            )
            if mode == "text_unit":
                candidate_labels = (
                    unit_label(candidate_words), quantity_label(candidate_words), time_label(candidate_words)
                )
                features[row_index, candidate_index, 2:] = tuple(
                    value
                    for expected, actual in zip(goal_labels, candidate_labels)
                    for value in match_state(expected, actual)
                )
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text", "text_unit"), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    train_bank, train_arrays = load_inputs("train", DATA)
    validation_bank, validation_arrays = load_inputs("validation", DATA)
    train_labels, _ = load_labels("train", train_bank, DATA)
    validation_labels, distinct = load_labels("validation", validation_bank, DATA)
    vocabulary, idf = fit_vocabulary(train_bank)
    train_emb, train_local = extract_feature_parts(train_bank, train_arrays["embeddings"])
    validation_emb, validation_local = extract_feature_parts(
        validation_bank, validation_arrays["embeddings"]
    )
    train_local = np.concatenate((train_local, goal_features(train_bank, args.mode, vocabulary, idf)), axis=-1)
    validation_local = np.concatenate((
        validation_local, goal_features(validation_bank, args.mode, vocabulary, idf)
    ), axis=-1)
    flat = train_local.reshape(-1, train_local.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    train_features = torch.from_numpy(combine_features(train_emb, train_local, mean, std))
    validation_features = torch.from_numpy(combine_features(validation_emb, validation_local, mean, std))
    informative = train_labels.sum(axis=1) < 4
    train_features = train_features[informative]
    train_labels = torch.from_numpy(train_labels[informative])
    validation_labels = torch.from_numpy(validation_labels)
    train_relations = torch.zeros((len(train_features), 4, 3), dtype=torch.float32)
    validation_relations = torch.zeros((len(validation_features), 4, 3), dtype=torch.float32)
    hard_mask = torch.from_numpy(distinct >= 2)
    states = []
    seed_metrics = []
    for seed in (17, 23, 42):
        state, metrics = train_one_seed(
            "independent", seed, train_features, train_relations, train_labels,
            validation_features, validation_relations, validation_labels, hard_mask,
            100, 15, 128, 1e-3, 1e-4, 0.0, "hard_loss",
        )
        states.append(state)
        seed_metrics.append({"seed": seed, **metrics})
    ensemble_metrics = evaluate_ensemble(
        "independent", states, validation_features, validation_relations,
        validation_labels, hard_mask, 0.0,
    )
    logits = torch.zeros((len(validation_bank), 4), dtype=torch.float32)
    with torch.inference_mode():
        for state in states:
            model = SetScorer(validation_features.shape[-1], "independent")
            model.load_state_dict(state)
            model.eval()
            logits += model(validation_features, validation_relations)
    selected = (logits / len(states)).argmax(dim=1).tolist()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions = {
        "protocol": PROTOCOL, "split": "validation",
        "ids": [row["id"] for row in validation_bank],
        "selected": [int(choice) for choice in selected],
    }
    args.output_prefix.with_name(args.output_prefix.name + "-validation-predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n"
    )
    metrics = {
        "mode": args.mode, "vocabulary_size": len(vocabulary),
        "seeds": seed_metrics, "ensemble_metrics": ensemble_metrics,
        "parameter_count": sum(p.numel() for p in SetScorer(validation_features.shape[-1], "independent").parameters()),
    }
    args.output_prefix.with_name(args.output_prefix.name + "-train.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics["ensemble_metrics"]))


if __name__ == "__main__":
    main()
