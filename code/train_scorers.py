"""Train parameter-matched independent or comparative candidate scorers."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import DATA, load_inputs, parse_answer, read_jsonl
from submission import (
    FEATURE_SCHEMA_VERSION,
    LOCAL_FEATURE_NAMES,
    PAIRWISE_RELATION_NAMES,
    RELATION_FEATURE_NAMES,
    SetScorer,
    combine_features,
    extract_feature_parts,
    extract_pairwise_relations,
    extract_relation_features,
    prepare_pairwise_relations,
    prepare_relation_features,
)


def load_labels(split, bank, data_dir):
    references = read_jsonl(Path(data_dir) / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference and candidate counts differ")

    labels = np.zeros((len(bank), 4), dtype=bool)
    distinct_valid_answers = np.zeros(len(bank), dtype=np.int64)
    for question_index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference and candidate rows are misaligned")
        gold = parse_answer(reference["answer"])
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        labels[question_index] = [answer is not None and answer == gold for answer in answers]
        distinct_valid_answers[question_index] = len(
            {answer for answer in answers if answer is not None}
        )
    if np.any(labels.sum(axis=1) == 0):
        raise ValueError("Every training group must have a correct candidate")
    return labels, distinct_valid_answers


def listwise_loss(logits, labels):
    log_probabilities = F.log_softmax(logits, dim=1)
    correct_log_probabilities = log_probabilities.masked_fill(~labels, -torch.inf)
    return -torch.logsumexp(correct_log_probabilities, dim=1).mean()


@torch.inference_mode()
def evaluate(model, features, relations, labels, hard_mask):
    model.eval()
    logits = model(features, relations)
    choices = logits.argmax(dim=1)
    selected_correct = labels.gather(1, choices[:, None]).squeeze(1)
    return {
        "full_accuracy": 100.0 * selected_correct.float().mean().item(),
        "hard_accuracy": 100.0 * selected_correct[hard_mask].float().mean().item(),
        "hard_loss": listwise_loss(logits[hard_mask], labels[hard_mask]).item(),
    }


def train_one_seed(
    context_mode,
    seed,
    train_features,
    train_relations,
    train_labels,
    validation_features,
    validation_relations,
    validation_labels,
    hard_mask,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    dropout,
    selection_metric,
):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = SetScorer(train_features.shape[-1], context_mode, dropout=dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = None
    best_metrics = None
    best_epoch = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_features[indices], train_relations[indices])
            loss = listwise_loss(logits, train_labels[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate(
            model,
            validation_features,
            validation_relations,
            validation_labels,
            hard_mask,
        )
        if selection_metric == "hard_loss":
            improved = (
                best_metrics is None
                or metrics["hard_loss"] < best_metrics["hard_loss"] - 1e-6
            )
        else:
            improved = best_metrics is None or (
                metrics["full_accuracy"],
                metrics["hard_accuracy"],
                -metrics["hard_loss"],
            ) > (
                best_metrics["full_accuracy"],
                best_metrics["hard_accuracy"],
                -best_metrics["hard_loss"] + 1e-6,
            )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or stale_epochs == 0:
            print(
                json.dumps(
                    {
                        "mode": context_mode,
                        "seed": seed,
                        "epoch": epoch,
                        **metrics,
                    }
                ),
                flush=True,
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate(
        model,
        validation_features,
        validation_relations,
        validation_labels,
        hard_mask,
    )
    final_metrics["best_epoch"] = best_epoch
    return best_state, final_metrics


@torch.inference_mode()
def evaluate_ensemble(
    context_mode, state_dicts, features, relations, labels, hard_mask, dropout
):
    logits = torch.zeros((len(features), 4), dtype=torch.float32)
    for state_dict in state_dicts:
        model = SetScorer(features.shape[-1], context_mode, dropout=dropout)
        model.load_state_dict(state_dict)
        model.eval()
        logits.add_(model(features, relations))
    logits.div_(len(state_dicts))
    choices = logits.argmax(dim=1)
    selected_correct = labels.gather(1, choices[:, None]).squeeze(1)
    return {
        "full_accuracy": 100.0 * selected_correct.float().mean().item(),
        "hard_accuracy": 100.0 * selected_correct[hard_mask].float().mean().item(),
        "hard_loss": listwise_loss(logits[hard_mask], labels[hard_mask]).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("independent", "comparative", "attention"),
        required=True,
    )
    parser.add_argument(
        "--relation-mode",
        choices=("none", "answer", "answer_cosine"),
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 42))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=("hard_loss", "full_accuracy"),
        default="hard_loss",
    )
    args = parser.parse_args()
    if args.model == "independent" and args.relation_mode != "none":
        parser.error("Independent models cannot use cross-candidate relations")
    if args.learning_rate <= 0.0:
        parser.error("learning rate must be positive")
    if args.weight_decay < 0.0:
        parser.error("weight decay cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")

    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train", args.data_dir)
    validation_bank, validation_arrays = load_inputs("validation", args.data_dir)
    train_labels_np, _ = load_labels("train", train_bank, args.data_dir)
    validation_labels_np, validation_distinct = load_labels(
        "validation", validation_bank, args.data_dir
    )

    train_embeddings, train_local = extract_feature_parts(
        train_bank, train_arrays["embeddings"]
    )
    validation_embeddings, validation_local = extract_feature_parts(
        validation_bank, validation_arrays["embeddings"]
    )
    flat_train_local = train_local.reshape(-1, train_local.shape[-1])
    local_mean = flat_train_local.mean(axis=0).astype(np.float32)
    local_std = flat_train_local.std(axis=0)
    local_std = np.where(local_std < 1e-6, 1.0, local_std).astype(np.float32)
    train_features_np = combine_features(
        train_embeddings, train_local, local_mean, local_std
    )
    validation_features_np = combine_features(
        validation_embeddings, validation_local, local_mean, local_std
    )
    if args.model == "attention":
        train_relations_np = prepare_pairwise_relations(
            extract_pairwise_relations(train_bank, train_embeddings),
            args.relation_mode,
        )
        validation_relations_np = prepare_pairwise_relations(
            extract_pairwise_relations(validation_bank, validation_embeddings),
            args.relation_mode,
        )
        relation_metadata = {
            "attention_version": 1,
            "pairwise_relation_names": list(PAIRWISE_RELATION_NAMES),
        }
    else:
        train_relations_raw = extract_relation_features(train_bank, train_embeddings)
        validation_relations_raw = extract_relation_features(
            validation_bank, validation_embeddings
        )
        flat_train_relations = train_relations_raw.reshape(
            -1, train_relations_raw.shape[-1]
        )
        relation_mean = flat_train_relations.mean(axis=0).astype(np.float32)
        relation_std = flat_train_relations.std(axis=0)
        relation_std = np.where(relation_std < 1e-6, 1.0, relation_std).astype(
            np.float32
        )
        train_relations_np = prepare_relation_features(
            train_relations_raw, relation_mean, relation_std, args.relation_mode
        )
        validation_relations_np = prepare_relation_features(
            validation_relations_raw,
            relation_mean,
            relation_std,
            args.relation_mode,
        )
        relation_metadata = {
            "relation_feature_names": list(RELATION_FEATURE_NAMES),
            "relation_mean": torch.from_numpy(relation_mean),
            "relation_std": torch.from_numpy(relation_std),
        }

    informative = train_labels_np.sum(axis=1) < 4
    train_features = torch.from_numpy(train_features_np[informative])
    train_relations = torch.from_numpy(train_relations_np[informative])
    train_labels = torch.from_numpy(train_labels_np[informative])
    validation_features = torch.from_numpy(validation_features_np)
    validation_relations = torch.from_numpy(validation_relations_np)
    validation_labels = torch.from_numpy(validation_labels_np)
    hard_mask = torch.from_numpy(validation_distinct >= 2)

    state_dicts = []
    seed_metrics = []
    for seed in args.seeds:
        state_dict, metrics = train_one_seed(
            args.model,
            seed,
            train_features,
            train_relations,
            train_labels,
            validation_features,
            validation_relations,
            validation_labels,
            hard_mask,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.dropout,
            args.selection_metric,
        )
        state_dicts.append(state_dict)
        seed_metrics.append({"seed": seed, **metrics})

    ensemble_metrics = evaluate_ensemble(
        args.model,
        state_dicts,
        validation_features,
        validation_relations,
        validation_labels,
        hard_mask,
        args.dropout,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in SetScorer(
            train_features.shape[-1], args.model, dropout=args.dropout
        ).parameters()
    )
    result = {
        "context_mode": args.model,
        "relation_mode": args.relation_mode,
        "seeds": list(args.seeds),
        "train_questions": len(train_bank),
        "informative_train_questions": int(informative.sum()),
        "validation_questions": len(validation_bank),
        "hard_validation_questions": int(hard_mask.sum()),
        "parameter_count": parameter_count,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "selection_metric": args.selection_metric,
        "seed_metrics": seed_metrics,
        "ensemble_metrics": ensemble_metrics,
        "training_seconds": time.perf_counter() - started,
    }
    bundle = {
        "format_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(LOCAL_FEATURE_NAMES),
        "context_mode": args.model,
        "relation_mode": args.relation_mode,
        "input_dim": int(train_features.shape[-1]),
        "local_mean": torch.from_numpy(local_mean),
        "local_std": torch.from_numpy(local_std),
        "dropout": args.dropout,
        "seeds": list(args.seeds),
        "state_dicts": state_dicts,
    }
    bundle.update(relation_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
