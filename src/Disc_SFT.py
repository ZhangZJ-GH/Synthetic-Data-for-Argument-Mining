import os

from am_prompts import SCHEMA_MAP, build_discriminator_prompt as build_prompt

import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# =========================
# Schema and sample normalization
# =========================


def normalize_framework(example_or_fw: Any) -> str:
    if isinstance(example_or_fw, dict):
        fw = example_or_fw.get("framework", "abstrct")
    else:
        fw = example_or_fw
    if fw is None:
        fw = "abstrct"
    return str(fw).strip().lower()


def extract_text_by_anchors(input_text: str, anchors: List[Dict[str, int]]) -> str:
    if not anchors:
        return ""
    pieces = []
    for a in anchors:
        try:
            start = int(a["from"])
            end = int(a["to"])
        except Exception:
            continue
        if start < 0 or end <= start or end > len(input_text):
            continue
        pieces.append(input_text[start:end])
    return " ".join(pieces).strip()


def convert_sample_to_text_node_format(
    example: Dict[str, Any],
    output_language_field: bool = True,
    keep_extra_top_fields: bool = False,
) -> Dict[str, Any]:
    """Convert a sample to the text-node representation used by this stage."""
    if "input" not in example:
        raise KeyError(f"Sample is missing the input field; available fields={list(example.keys())}")

    input_text = str(example["input"])
    framework = normalize_framework(example)

    nodes = []
    for n in example.get("nodes", []):
        if not isinstance(n, dict):
            continue
        if "id" not in n or "label" not in n:
            continue

        if "text" in n:
            node_text = str(n["text"]).strip()
        else:
            node_text = extract_text_by_anchors(input_text, n.get("anchors", []))

        nodes.append(
            {
                "id": int(n["id"]),
                "label": str(n["label"]).strip(),
                "text": node_text,
            }
        )

    edges = []
    for e in example.get("edges", []):
        if not isinstance(e, dict):
            continue
        if "source" not in e or "target" not in e or "label" not in e:
            continue
        edges.append(
            {
                "source": int(e["source"]),
                "target": int(e["target"]),
                "label": str(e["label"]).strip(),
            }
        )

    out = {
        "input": input_text,
        "framework": framework,
        "nodes": sorted(nodes, key=lambda x: x["id"]),
        "edges": sorted(edges, key=lambda x: (x["source"], x["target"], x["label"])),
        "tops": sorted([int(x) for x in example.get("tops", [])]),
    }

    if output_language_field and "language" in example:
        out["language"] = example["language"]

    if keep_extra_top_fields:
        reserved = {"input", "framework", "nodes", "edges", "tops", "language"}
        for k, v in example.items():
            if k not in reserved:
                out[k] = v

    return out


def serialize_sample_for_discriminator(
    example: Dict[str, Any],
    output_language_field: bool = True,
    keep_extra_top_fields: bool = False,
    indent: int = 2,
) -> str:
    normalized = convert_sample_to_text_node_format(
        example,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
    )
    return json.dumps(normalized, ensure_ascii=False, indent=indent, sort_keys=False)


# =========================
# Utility functions
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL: {path}, line {line_id + 1}\n{e}")
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def group_synthetic_by_real(
    real_data: List[Dict[str, Any]],
    synthetic_data: List[Dict[str, Any]],
    output_language_field: bool,
    keep_extra_top_fields: bool,
) -> List[Dict[str, Any]]:
    """Pair each real sample with its three consecutive synthetic samples."""
    if len(synthetic_data) != len(real_data) * 3:    # Fixed candidate count
        raise ValueError(
            f"Synthetic record count mismatch: len(synthetic)={len(synthetic_data)}, "
            f"expected len(real)*3={len(real_data) * 3}"
        )

    groups = []
    for i, real_item in enumerate(real_data):
        real_text = serialize_sample_for_discriminator(
            real_item,
            output_language_field=output_language_field,
            keep_extra_top_fields=keep_extra_top_fields,
        )

        synth_items = synthetic_data[i * 3: (i + 1) * 3]   # Fixed candidate count
        synth_texts = [
            serialize_sample_for_discriminator(
                x,
                output_language_field=output_language_field,
                keep_extra_top_fields=keep_extra_top_fields,
            )
            for x in synth_items
        ]

        groups.append({
            "group_id": i,
            "real_sample_text": real_text,
            "synthetic_sample_texts": synth_texts,
        })
    return groups


def build_examples_from_group(
    group: Dict[str, Any],
    num_permutations: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Construct distinct four-way candidate permutations for one group."""
    base = [
        {"text": group["real_sample_text"], "is_real": 1, "source": "real"},
        {"text": group["synthetic_sample_texts"][0], "is_real": 0, "source": "synthetic_1"},
        {"text": group["synthetic_sample_texts"][1], "is_real": 0, "source": "synthetic_2"},
        {"text": group["synthetic_sample_texts"][2], "is_real": 0, "source": "synthetic_3"},
    ]   # Fixed candidate count

    seen_orders = set()
    examples = []

    max_trials = max(50, num_permutations * 10)
    trials = 0
    while len(examples) < num_permutations and trials < max_trials:
        trials += 1
        order = list(range(4))
        rng.shuffle(order)
        order = tuple(order)
        if order in seen_orders:
            continue
        seen_orders.add(order)

        shuffled = [base[idx] for idx in order]
        texts = [x["text"] for x in shuffled]
        label = next(i for i, x in enumerate(shuffled) if x["is_real"] == 1)

        prompt = build_prompt(texts)

        examples.append({
            "group_id": group["group_id"],   # Fixed candidate count
            "prompt": prompt,
            "candidates": {
                "A": texts[0],
                "B": texts[1],
                "C": texts[2],
                "D": texts[3],
            },
            "label": label,
            "real_position": ["A", "B", "C", "D"][label],
        })

    return examples


def split_groups(
    groups: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split source groups before permutations to avoid within-split group leakage."""
    indices = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:val_size])

    train_groups, val_groups = [], []
    for i, g in enumerate(groups):
        if i in val_indices:
            val_groups.append(g)
        else:
            train_groups.append(g)
    return train_groups, val_groups


# =========================
# Dataset
# =========================

class DiscriminatorDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["prompt"],
            truncation=False,
            padding=False,
        )
        check_token_length(len(enc["input_ids"]), self.max_length, f"Discriminator example {idx}")
        enc["labels"] = ex["label"]
        return enc


# =========================
# Model helpers
# =========================

def find_modules_to_save_for_cls(model) -> List[str]:
    modules = []
    if hasattr(model, "score"):
        modules.append("score")
    if hasattr(model, "classifier"):
        modules.append("classifier")
    if hasattr(model, "classifier_head"):
        modules.append("classifier_head")
    return list(sorted(set(modules)))


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=4,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.id2label = {0: "A", 1: "B", 2: "C", 3: "D"}   # Fixed candidate count
    model.config.label2id = {"A": 0, "B": 1, "C": 2, "D": 3}   # Fixed candidate count

    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        modules_to_save = find_modules_to_save_for_cls(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.target_modules.split(","),
            modules_to_save=modules_to_save if len(modules_to_save) > 0 else None,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


# =========================
# Evaluation metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()

    pred_counts = {k: 0 for k in ["A", "B", "C", "D"]}
    for p in preds:
        pred_counts[["A", "B", "C", "D"][int(p)]] += 1

    return {
        "accuracy": acc,
        "pred_A": pred_counts["A"],
        "pred_B": pred_counts["B"],
        "pred_C": pred_counts["C"],
        "pred_D": pred_counts["D"],
    }


# =========================
# Scoring helper for reinforcement learning
# =========================

@torch.no_grad()
def score_candidate_group(model, tokenizer, candidates: List[str], max_length: int, device: str):
    max_length = resolve_token_budget(tokenizer, model, max_length)
    prompt = build_prompt(candidates)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )
    check_token_length(int(inputs["input_ids"].shape[1]), max_length, "Discriminator scoring group")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)

    return {
        "logits": logits.detach().cpu().tolist(),
        "probs": probs.detach().cpu().tolist(),
        "pred_index": int(torch.argmax(probs).item()),
        "pred_label": ["A", "B", "C", "D"][int(torch.argmax(probs).item())],
    }


# =========================
# Main entry point
# =========================


def resolve_token_budget(tokenizer, model, requested: int) -> int:
    """Respect both the configured budget and the model's context window."""
    if requested <= 0:
        raise ValueError("Token budgets must be positive.")
    limits = [int(requested)]
    config = getattr(model, "config", None)
    for value in (getattr(config, "max_position_embeddings", None),
                  getattr(tokenizer, "model_max_length", None)):
        if isinstance(value, int) and 0 < value < 1_000_000:
            limits.append(value)
    return min(limits)


def check_token_length(length: int, budget: int, context: str) -> None:
    """Never silently truncate a candidate, reference, or supervised target."""
    if length > budget:
        raise ValueError(
            f"{context} needs {length} tokens, but the available budget is {budget}. "
            "No truncation was applied. Increase the configured budget within the "
            "model context limit, or use a model/data sample that fits."
        )


def validate_text_lengths(examples, tokenizer, budget: int, context: str) -> None:
    longest = 0
    for index, example in enumerate(examples):
        length = len(tokenizer(example["prompt"], truncation=False, padding=False)["input_ids"])
        check_token_length(length, budget, f"{context} example {index}")
        longest = max(longest, length)
    print(f"[Length check] {context}: examples={len(examples)}, longest={longest}, budget={budget}")


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--real_file", type=str, required=True)
    parser.add_argument("--synthetic_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="input", help="Retained for compatibility with older scripts; unused in full-sample discrimination.")

    # full-sample normalization
    parser.add_argument("--output_language_field", action="store_true")
    parser.add_argument("--keep_extra_top_fields", action="store_true")

    # model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=32768)

    # data building
    parser.add_argument("--train_permutations_per_group", type=int, default=6)
    parser.add_argument("--eval_permutations_per_group", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")

    # LoRA / QLoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names. Adjust them for a different model architecture.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    real_data = load_jsonl(args.real_file)
    synthetic_data = load_jsonl(args.synthetic_file)

    print(f"Data loaded: real={len(real_data)}, synthetic={len(synthetic_data)}")

    groups = group_synthetic_by_real(
        real_data=real_data,
        synthetic_data=synthetic_data,
        output_language_field=args.output_language_field,
        keep_extra_top_fields=args.keep_extra_top_fields,
    )
    print(f"Number of aligned groups: {len(groups)}")

    train_groups, val_groups = split_groups(groups, args.val_ratio, args.seed)
    print(f"train groups={len(train_groups)}, val groups={len(val_groups)}")

    rng_train = random.Random(args.seed)
    rng_val = random.Random(args.seed + 1)

    train_examples = []
    for g in train_groups:
        train_examples.extend(
            build_examples_from_group(
                g,
                num_permutations=args.train_permutations_per_group,
                rng=rng_train,
            )
        )

    val_examples = []
    for g in val_groups:
        val_examples.extend(
            build_examples_from_group(
                g,
                num_permutations=args.eval_permutations_per_group,
                rng=rng_val,
            )
        )

    print(f"train examples={len(train_examples)}, val examples={len(val_examples)}")

    save_jsonl(os.path.join(args.output_dir, "train_examples_preview.jsonl"), train_examples[:200])
    save_jsonl(os.path.join(args.output_dir, "val_examples_preview.jsonl"), val_examples[:200])

    model, tokenizer = build_model_and_tokenizer(args)
    args.max_length = resolve_token_budget(tokenizer, model, args.max_length)
    validate_text_lengths(train_examples, tokenizer, args.max_length, "Discriminator train")
    validate_text_lengths(val_examples, tokenizer, args.max_length, "Discriminator validation")

    train_dataset = DiscriminatorDataset(train_examples, tokenizer, args.max_length)
    val_dataset = DiscriminatorDataset(val_examples, tokenizer, args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=(not args.bf16 and torch.cuda.is_available()),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_args_used.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Training completed. Model saved to: {args.output_dir}")

    if len(val_examples) > 0:
        sample = val_examples[0]
        candidates = [
            sample["candidates"]["A"],
            sample["candidates"]["B"],
            sample["candidates"]["C"],
            sample["candidates"]["D"],
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.load_in_4bit:
            model.to(device)

        result = score_candidate_group(
            model=model,
            tokenizer=tokenizer,
            candidates=candidates,
            max_length=args.max_length,
            device=device,
        )

        demo = {
            "real_position": sample["real_position"],
            "model_output": result
        }
        with open(os.path.join(args.output_dir, "demo_score.json"), "w", encoding="utf-8") as f:
            json.dump(demo, f, ensure_ascii=False, indent=2)

        print("Saved demo_score.json with four-class logits, probabilities, and the predicted index.")


if __name__ == "__main__":
    main()
