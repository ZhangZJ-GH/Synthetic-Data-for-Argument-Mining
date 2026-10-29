# -*- coding: utf-8 -*-

"""Supervised training from labeled source samples to paired synthetic targets.
All source and target records are consumed in file order; index pairing
requires an equal number of records in the two files."""

import os

from am_prompts import SCHEMA_MAP, get_schema, build_system_prompt, build_user_prompt

import json
import uuid
import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig



def parse_args():
    parser = argparse.ArgumentParser()

    # Node and edge count settings
    parser.add_argument("--min_nodes", type=int)
    parser.add_argument("--min_edges", type=int)

    # Data files
    parser.add_argument("--train_file", type=str, required=True, help="Source training dataset in JSONL format")
    parser.add_argument("--synthetic_train_file", type=str, required=True, help="Paired manually written synthetic training targets in JSONL format")

    parser.add_argument("--valid_file", type=str, default=None, help="Optional source validation dataset in JSONL format")
    parser.add_argument("--synthetic_valid_file", type=str, default=None, help="Optional paired manually written synthetic validation targets in JSONL format")

    parser.add_argument("--pair_mode", type=str, default="index", choices=["index", "id"])

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Output format settings
    parser.add_argument("--output_language_field", action="store_true")
    parser.add_argument(
        "--keep_extra_top_fields",
        action="store_true",
        help="Keep extra top-level fields in completions beyond input/framework/language/nodes/edges/tops",
    )

    # Training parameters
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # ========= LoRA / QLoRA =========
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON in {path}, line {line_id}: {e}")
    return data





def normalize_framework(example: Dict[str, Any]) -> str:
    fw = example.get("framework", "abstrct")
    if fw is None:
        fw = "abstrct"
    return str(fw).strip().lower()


def extract_text_by_anchors(input_text: str, anchors: List[Dict[str, int]]) -> str:
    if not anchors:
        return ""
    pieces = []
    for a in anchors:
        start = int(a["from"])
        end = int(a["to"])
        pieces.append(input_text[start:end])
    return " ".join(pieces).strip()


def find_substring_indices(input_text: str, substring: str) -> Optional[Dict[str, int]]:
    substring = substring.strip()
    if not substring:
        return None
    start = input_text.find(substring)
    if start == -1:
        return None
    return {"from": start, "to": start + len(substring)}


def detect_bf16_support() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def convert_sample_to_text_node_format(
    example: Dict[str, Any],
    output_language_field: bool = False,
    keep_extra_top_fields: bool = False,
) -> Dict[str, Any]:
    """Convert a sample to the text-node representation used by this stage."""
    input_text = example["input"]
    framework = normalize_framework(example)

    nodes = []
    for n in example.get("nodes", []):
        node_text = extract_text_by_anchors(input_text, n.get("anchors", []))
        nodes.append(
            {
                "id": n["id"],
                "label": n["label"],
                "text": node_text,
            }
        )

    edges = []
    for e in example.get("edges", []):
        edges.append(
            {
                "source": e["source"],
                "target": e["target"],
                "label": e["label"],
            }
        )

    out = {
        "input": input_text,
        "framework": framework,
        "nodes": nodes,
        "edges": edges,
        "tops": example.get("tops", []),
    }

    if output_language_field and "language" in example:
        out["language"] = example["language"]

    if keep_extra_top_fields:
        reserved = {"input", "framework", "nodes", "edges", "tops", "language"}
        for k, v in example.items():
            if k not in reserved:
                out[k] = v

    return out


def recover_anchors_from_text_nodes(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Recover anchor-node annotations from generated text-node annotations."""
    input_text = sample["input"]
    framework = sample["framework"]

    new_nodes = []
    for node in sample.get("nodes", []):
        text = node.get("text", "").strip()
        anchor = find_substring_indices(input_text, text)
        if anchor is None:
            continue
        new_nodes.append(
            {
                "id": node["id"],
                "label": node["label"],
                "anchors": [anchor],
            }
        )

    valid_ids = {n["id"] for n in new_nodes}

    new_edges = [
        e for e in sample.get("edges", [])
        if e["source"] in valid_ids and e["target"] in valid_ids
    ]
    new_tops = [t for t in sample.get("tops", []) if t in valid_ids]

    out = {
        "id": f"synthetic_{uuid.uuid4().hex[:12]}",
        "input": input_text,
        "framework": framework,
        "nodes": new_nodes,
        "edges": new_edges,
        "tops": new_tops,
        "source": "synthetic_qwen3_sft",
        "provenance": "synthetic_qwen3_sft",
        "version": 1.0,
    }

    if "language" in sample:
        out["language"] = sample["language"]

    return out


def pair_datasets(
    src_data: List[Dict[str, Any]],
    tgt_data: List[Dict[str, Any]],
    pair_mode: str,
) -> List[Dict[str, Any]]:
    """Return source/target pairs using file order or matching top-level IDs."""
    if pair_mode == "index":
        if len(src_data) != len(tgt_data):
            raise ValueError(
                f"pair_mode=index requires equal record counts, but got {len(src_data)} vs {len(tgt_data)}"
            )
        pairs = []
        for s, t in zip(src_data, tgt_data):
            pairs.append({"source": s, "target": t})
        return pairs

    # pair_mode == "id"
    tgt_map = {}
    for ex in tgt_data:
        if "id" not in ex:
            raise ValueError("pair_mode=id requires a top-level id field in every synthetic record")
        tgt_map[str(ex["id"])] = ex

    pairs = []
    missing = []
    for s in src_data:
        if "id" not in s:
            raise ValueError("pair_mode=id requires a top-level id field in every source record")
        sid = str(s["id"])
        if sid not in tgt_map:
            missing.append(sid)
            continue
        pairs.append({"source": s, "target": tgt_map[sid]})

    if missing:
        raise ValueError(f"pair_mode=id found no synthetic targets for these source IDs: {missing[:10]}")

    return pairs

def estimate_min_input_words(min_nodes: int, framework: str) -> int:
    # Word estimates assume English; use characters or tokenizer tokens for other languages.
    return max(80, int(min_nodes) * 12)


def preprocess_pair(
    pair: Dict[str, Dict[str, Any]],
    output_language_field: bool,
    keep_extra_top_fields: bool,
    min_nodes: int,
    min_edges: int,
) -> Dict[str, Any]:
    source_raw = pair["source"]
    target_raw = pair["target"]

    framework = normalize_framework(source_raw)

    reference_sample = convert_sample_to_text_node_format(
        source_raw,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
    )

    target_sample = convert_sample_to_text_node_format(
        target_raw,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
    )

    system_prompt = build_system_prompt(
        framework=framework,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
        min_nodes=min_nodes,
        min_edges=min_edges,
    )
    user_prompt = build_user_prompt(
        reference_sample=reference_sample,
        framework=framework,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
        min_nodes=min_nodes,
        min_edges=min_edges,
    )

    # prompt = open(f"prompt/{framework}_Gener.txt").read()
    # user_prompt = prompt.replace(
    #     "<<<REFERENCE_JSON>>>",
    #     # json.dumps(reference_sample, ensure_ascii=False, indent=2)
    #     json.dumps(source_raw, ensure_ascii=False)
    # )

    completion_text = json.dumps(target_sample, ensure_ascii=False, separators=(",", ":"))
    # completion_text = json.dumps(target_raw, ensure_ascii=False)

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "completion": [
            {"role": "assistant", "content": completion_text},
        ],
    }


def build_dataset(
    source_file: str,
    target_file: str,
    pair_mode: str,
    output_language_field: bool,
    keep_extra_top_fields: bool,
    min_nodes: int,
    min_edges: int,
) -> Dataset:
    src_data = read_jsonl(source_file)

    # Read every target in file order; index pairing requires equal file lengths.
    tgt_data = read_jsonl(target_file)

    pairs = pair_datasets(src_data, tgt_data, pair_mode=pair_mode)

    processed = []
    for pair in pairs:
        processed.append(
            preprocess_pair(
                pair=pair,
                output_language_field=output_language_field,
                keep_extra_top_fields=keep_extra_top_fields,
                min_nodes=min_nodes,
                min_edges=min_edges,
            )
        )

    return Dataset.from_list(processed)


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str, use_4bit: bool, use_bf16: bool):
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    return model


def build_lora_config(args) -> LoraConfig:
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def print_one_sample(dataset: Dataset, title: str):
    if len(dataset) == 0:
        print(f"{title}: empty dataset")
        return

    sample = dataset[0]
    print("=" * 100)
    print(title)
    print("-" * 100)
    print("PROMPT:")
    print(json.dumps(sample["prompt"], ensure_ascii=False, indent=2))
    print("-" * 100)
    print("COMPLETION:")
    print(json.dumps(sample["completion"], ensure_ascii=False, indent=2))
    print("=" * 100)



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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    use_bf16 = detect_bf16_support()
    print(f"bf16 supported: {use_bf16}")
    print(f"Using 4-bit QLoRA: {args.load_in_4bit}")

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(
        args.model_path,
        use_4bit=args.load_in_4bit,
        use_bf16=use_bf16,
    )
    lora_config = build_lora_config(args)

    train_dataset = build_dataset(
        source_file=args.train_file,
        target_file=args.synthetic_train_file,
        pair_mode=args.pair_mode,
        output_language_field=args.output_language_field,
        keep_extra_top_fields=args.keep_extra_top_fields,
        min_nodes=args.min_nodes,
        min_edges=args.min_edges,
    )

    eval_dataset = None
    if args.valid_file and args.synthetic_valid_file:
        eval_dataset = build_dataset(
            source_file=args.valid_file,
            target_file=args.synthetic_valid_file,
            pair_mode=args.pair_mode,
            output_language_field=args.output_language_field,
            keep_extra_top_fields=args.keep_extra_top_fields,
            min_nodes=args.min_nodes,
            min_edges=args.min_edges,
        )

    print_one_sample(train_dataset, "TRAIN SAMPLE AFTER PREPROCESS")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=use_bf16,
        fp16=not use_bf16,
        # tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Keep full sequences; validate their token lengths before training.
        max_length=None,
        packing=False,
        report_to="none",
        remove_unused_columns=False,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        completion_only_loss=True,
        assistant_only_loss=False,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )


    args.max_length = resolve_token_budget(tokenizer, model, args.max_length)
    for split_name, prepared_dataset in (("train", trainer.train_dataset),
                                         ("validation", trainer.eval_dataset)):
        if prepared_dataset is None:
            continue
        longest = 0
        for index, example in enumerate(prepared_dataset):
            length = len(example["input_ids"])
            check_token_length(length, args.max_length, f"SFT {split_name} example {index}")
            longest = max(longest, length)
        print(f"[Length check] SFT {split_name}: longest={longest}, budget={args.max_length}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Training completed. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
