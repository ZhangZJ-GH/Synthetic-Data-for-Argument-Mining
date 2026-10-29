# -*- coding: utf-8 -*-

import os

from am_prompts import (
    SCHEMA_MAP, get_schema, build_system_prompt, build_user_prompt,
    build_discriminator_prompt,
)

import re
import json
import math
import copy
import random
import argparse
import time
import gc
import sys
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    set_seed,
)

from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = hasattr(torch, "npu")
except Exception:
    HAS_TORCH_NPU = False


# =========================================================
# Runtime utilities with GPU/CUDA support
# =========================================================

def log_info(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def init_log_file(output_dir: str, filename: str = "train_terminal.log"):
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, filename)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    start_banner = (
        "=" * 120 + "\n"
        + f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + f"log_file: {log_path}\n"
        + "=" * 120 + "\n"
    )
    log_f.write(start_banner)
    log_f.flush()
    sys.stdout = TeeStream(sys.__stdout__, log_f)
    sys.stderr = TeeStream(sys.__stderr__, log_f)
    return log_path, log_f


def should_disable_internal_tee(args) -> bool:
    if getattr(args, "disable_internal_tee", False):
        return True
    env_value = os.environ.get("DISABLE_INTERNAL_TEE", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSON: {path}, line {line_id}\n{e}")
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    """Atomic JSON save for lightweight training/round state."""
    ensure_dir(os.path.dirname(path) or ".")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def is_complete_lora_adapter_dir(path: Optional[str]) -> bool:
    """Return True when a directory looks like a fully saved PEFT/LoRA adapter."""
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        return False
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in weight_names)


def _checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path.rstrip(os.sep)))
    if not m:
        return -1
    return int(m.group(1))


def find_latest_trainer_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest HuggingFace/TRL Trainer checkpoint under output_dir."""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        if _checkpoint_step(path) < 0:
            continue
        # trainer_state.json is the most useful marker for true train-state resume.
        # Some PEFT checkpoints may not include every file, but still can restore model weights.
        if (
            os.path.exists(os.path.join(path, "trainer_state.json"))
            or is_complete_lora_adapter_dir(path)
        ):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=_checkpoint_step)
    return candidates[-1]


def resolve_grpo_resume_checkpoint(args, round_out: str) -> Optional[str]:
    """Resolve --grpo_resume_from_checkpoint. Values: auto, none, or a checkpoint path."""
    mode = str(getattr(args, "grpo_resume_from_checkpoint", "auto") or "auto").strip()
    if mode.lower() in {"none", "false", "0", "no", "off"}:
        return None
    if mode.lower() == "auto":
        return find_latest_trainer_checkpoint(round_out)
    if not os.path.isdir(mode):
        raise FileNotFoundError(f"Specified GRPO checkpoint directory does not exist: {mode}")
    return mode


def get_replay_buffer_path(args) -> str:
    return os.path.join(args.output_dir, "replay_buffer_state.jsonl")


def save_replay_buffer_state(args, replay_buffer) -> None:
    path = get_replay_buffer_path(args)
    save_jsonl(path, getattr(replay_buffer, "buffer", []))
    log_info(f"Replay buffer state saved: {path} | size={len(replay_buffer)}")


def load_replay_buffer_state_if_available(args, replay_buffer) -> None:
    path = get_replay_buffer_path(args)
    if os.path.exists(path):
        replay_buffer.buffer = load_jsonl(path)
        log_info(f"Replay buffer state restored: {path} | size={len(replay_buffer)}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_npu_available() -> bool:
    return HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()





def get_default_device() -> str:
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_runtime_device(explicit_device: Optional[str] = None) -> str:
    return explicit_device if explicit_device is not None else get_default_device()


def clear_accelerator_cache() -> None:
    if is_npu_available():
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    elif is_cuda_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def synchronize_accelerator() -> None:
    if is_npu_available():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    elif is_cuda_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def collect_accelerator_ipc() -> None:
    if is_cuda_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def npu_guard_4bit(load_in_4bit: bool, component_name: str) -> bool:
    if load_in_4bit and is_npu_available():
        log_info(f"[Warning] {component_name}: bitsandbytes 4-bit is not supported on this backend; automatically disabling load_in_4bit.")
        return False
    return load_in_4bit


def choose_torch_dtype(bf16: bool):
    return torch.bfloat16 if bf16 else torch.float16


def maybe_build_bnb_config(load_in_4bit: bool, bf16: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=choose_torch_dtype(bf16),
        bnb_4bit_use_double_quant=True,
    )


def completion_to_text(x):
    """Normalize TRL completions represented as strings or message lists."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        texts = []
        for item in x:
            if isinstance(item, dict) and "content" in item:
                texts.append(str(item["content"]))
            else:
                texts.append(str(item))
        return "\n".join(texts).strip()
    return str(x).strip()


# =========================================================
# Generator schema, prompts, and discriminator serialization
# =========================================================



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
        start = int(a["from"])
        end = int(a["to"])
        if start < 0 or end <= start or end > len(input_text):
            continue
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


def convert_sample_to_text_node_format(
    example: Dict[str, Any],
    output_language_field: bool = False,
    keep_extra_top_fields: bool = False,
) -> Dict[str, Any]:
    """Convert a sample to the text-node representation used by this stage."""
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
    output_language_field: bool = False,
    keep_extra_top_fields: bool = False,
    indent: int = 2,
) -> str:
    normalized = convert_sample_to_text_node_format(
        example,
        output_language_field=output_language_field,
        keep_extra_top_fields=keep_extra_top_fields,
    )
    return json.dumps(normalized, ensure_ascii=False, indent=indent, sort_keys=False)

def estimate_min_input_words(min_nodes: int, framework: str) -> int:
    # Word estimates assume English; use characters or tokenizer tokens for other languages.
    return max(80, int(min_nodes) * 12)

def build_generator_messages_from_source(
    source_raw: Dict[str, Any],
    output_language_field: bool,
    keep_extra_top_fields: bool,
    min_nodes: int,
    min_edges: int,
) -> Tuple[List[Dict[str, str]], str]:
    framework = normalize_framework(source_raw)

    reference_sample = convert_sample_to_text_node_format(
        source_raw,
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, framework


def render_generator_prompt(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    text = ""
    for m in messages:
        text += f"[{m['role'].upper()}]\n{m['content']}\n\n"
    text += "[ASSISTANT]\n"
    return text


# =========================================================
# Qwen3 reasoning cleanup, JSON extraction, and validation
# =========================================================

THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    cleaned = THINK_PATTERN.sub("", text)
    first_brace = cleaned.find("{")
    if first_brace != -1:
        cleaned = cleaned[first_brace:]
    return cleaned.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def sanitize_generated_sample(
    sample: Dict[str, Any],
    expected_framework: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(sample, dict):
        return None

    if "input" not in sample:
        return None

    input_text = str(sample["input"]).strip()
    if not input_text:
        return None

    framework = normalize_framework(sample.get("framework", expected_framework or "abstrct"))
    if expected_framework is not None and framework != expected_framework:
        framework = expected_framework

    raw_nodes = sample.get("nodes", [])
    raw_edges = sample.get("edges", [])
    raw_tops = sample.get("tops", [])

    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []
    if not isinstance(raw_tops, list):
        raw_tops = []

    new_nodes = []
    valid_ids = set()

    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        if "id" not in node or "label" not in node or "text" not in node:
            continue
        try:
            nid = int(node["id"])
        except Exception:
            continue

        label = str(node["label"]).strip()
        text = str(node["text"]).strip()

        if not label or not text:
            continue
        if find_substring_indices(input_text, text) is None:
            continue

        new_nodes.append({
            "id": nid,
            "label": label,
            "text": text,
        })
        valid_ids.add(nid)

    if len(new_nodes) == 0:
        return None

    new_edges = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        if "source" not in edge or "target" not in edge or "label" not in edge:
            continue
        try:
            s = int(edge["source"])
            t = int(edge["target"])
        except Exception:
            continue
        if s not in valid_ids or t not in valid_ids:
            continue

        new_edges.append({
            "source": s,
            "target": t,
            "label": str(edge["label"]).strip(),
        })

    new_tops = []
    for top in raw_tops:
        try:
            tid = int(top)
        except Exception:
            continue
        if tid in valid_ids:
            new_tops.append(tid)

    if len(new_tops) == 0 and len(new_nodes) > 0:
        new_tops = [new_nodes[0]["id"]]

    out = {
        "input": input_text,
        "framework": framework,
        "nodes": sorted(new_nodes, key=lambda x: x["id"]),
        "edges": sorted(new_edges, key=lambda x: (x["source"], x["target"], x["label"])),
        "tops": sorted(new_tops),
    }

    if "language" in sample:
        out["language"] = sample["language"]

    return out


def parse_qwen_generated_json(
    raw_completion: Any,
    expected_framework: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    raw_text = completion_to_text(raw_completion)
    cleaned = strip_think_blocks(raw_text)
    json_str = extract_first_json_object(cleaned)
    if json_str is None:
        return None

    try:
        obj = json.loads(json_str)
    except Exception:
        return None

    return sanitize_generated_sample(obj, expected_framework=expected_framework)


def parse_qwen_generated_json_with_stats(
    raw_completion: Any,
    expected_framework: Optional[str] = None,
) -> Dict[str, Any]:
    raw_text = completion_to_text(raw_completion)
    cleaned = strip_think_blocks(raw_text)
    json_str = extract_first_json_object(cleaned)
    if json_str is None:
        return {
            "json_str": None,
            "raw_obj": None,
            "sample": None,
            "num_raw_nodes_with_text": 0,
            "num_exact_match_nodes": 0,
            "exact_match_ratio": 0.0,
        }

    try:
        obj = json.loads(json_str)
    except Exception:
        return {
            "json_str": json_str,
            "raw_obj": None,
            "sample": None,
            "num_raw_nodes_with_text": 0,
            "num_exact_match_nodes": 0,
            "exact_match_ratio": 0.0,
        }

    input_text = str(obj.get("input", "")).strip()
    raw_nodes = obj.get("nodes", []) if isinstance(obj.get("nodes", []), list) else []
    num_raw_nodes_with_text = 0
    num_exact_match_nodes = 0

    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_text = str(node.get("text", "")).strip()
        if not node_text:
            continue
        num_raw_nodes_with_text += 1
        if input_text and find_substring_indices(input_text, node_text) is not None:
            num_exact_match_nodes += 1

    exact_match_ratio = (
        float(num_exact_match_nodes) / float(num_raw_nodes_with_text)
        if num_raw_nodes_with_text > 0 else 0.0
    )

    sanitized = sanitize_generated_sample(obj, expected_framework=expected_framework)
    return {
        "json_str": json_str,
        "raw_obj": obj,
        "sample": sanitized,
        "num_raw_nodes_with_text": num_raw_nodes_with_text,
        "num_exact_match_nodes": num_exact_match_nodes,
        "exact_match_ratio": exact_match_ratio,
    }


def build_fixed_anchor_fake_texts(
    real_sample_for_disc_text: str,
    output_language_field: bool = False,
    keep_extra_top_fields: bool = False,
) -> List[str]:
    try:
        real_obj = json.loads(real_sample_for_disc_text)
    except Exception:
        return [real_sample_for_disc_text, real_sample_for_disc_text]

    anchor_1 = copy.deepcopy(real_obj)
    anchor_2 = copy.deepcopy(real_obj)

    nodes_1 = anchor_1.get("nodes", []) if isinstance(anchor_1.get("nodes", []), list) else []
    if len(nodes_1) > 0 and isinstance(nodes_1[0], dict):
        first_text = str(nodes_1[0].get("text", "")).strip()
        nodes_1[0]["text"] = (first_text + " __span_mismatch__").strip() if first_text else "__span_mismatch__"
    anchor_1["nodes"] = nodes_1
    anchor_1["edges"] = []
    if len(nodes_1) > 0 and isinstance(nodes_1[0], dict) and "id" in nodes_1[0]:
        try:
            anchor_1["tops"] = [int(nodes_1[0]["id"])]
        except Exception:
            anchor_1["tops"] = [0]

    nodes_2 = anchor_2.get("nodes", []) if isinstance(anchor_2.get("nodes", []), list) else []
    if len(nodes_2) > 0 and all(isinstance(n, dict) for n in nodes_2):
        if len(nodes_2) >= 2:
            text_a = str(nodes_2[0].get("text", "")).strip()
            text_b = str(nodes_2[1].get("text", "")).strip()
            nodes_2[0]["text"] = text_b
            nodes_2[1]["text"] = text_a
        else:
            only_text = str(nodes_2[0].get("text", "")).strip()
            nodes_2[0]["text"] = ("swapped_" + only_text) if only_text else "swapped_text"
    anchor_2["nodes"] = nodes_2
    if isinstance(anchor_2.get("edges", []), list) and len(anchor_2["edges"]) > 1:
        anchor_2["edges"] = list(reversed(anchor_2["edges"]))

    return [
        serialize_sample_for_discriminator(anchor_1, output_language_field=output_language_field, keep_extra_top_fields=keep_extra_top_fields),
        serialize_sample_for_discriminator(anchor_2, output_language_field=output_language_field, keep_extra_top_fields=keep_extra_top_fields),
    ]


def exact_match_rule_reward(exact_match_ratio: float, has_any_node_text: bool) -> float:
    if not has_any_node_text:
        return -1.0
    return 2.0 * float(exact_match_ratio) - 1.0


# =========================================================
# Discriminator prompt
# =========================================================

def load_trainable_generator(
    base_model_path: str,
    lora_path: Optional[str],
    bf16: bool,
    load_in_4bit: bool,
):
    load_in_4bit = npu_guard_4bit(load_in_4bit, "generator(trainable)")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=choose_torch_dtype(bf16),
        quantization_config=maybe_build_bnb_config(load_in_4bit, bf16),
        device_map="auto" if load_in_4bit and not is_npu_available() else None,
    )

    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    model.config.use_cache = False
    return model, tokenizer


def load_eval_generator(
    base_model_path: str,
    lora_path: Optional[str],
    bf16: bool,
    load_in_4bit: bool,
):
    load_in_4bit = npu_guard_4bit(load_in_4bit, "generator(eval)")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=choose_torch_dtype(bf16),
        quantization_config=maybe_build_bnb_config(load_in_4bit, bf16),
        device_map="auto" if load_in_4bit and not is_npu_available() else None,
    )
    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    model.eval()
    return model, tokenizer


def load_trainable_discriminator(
    base_model_path: str,
    lora_path: Optional[str],
    bf16: bool,
    load_in_4bit: bool,
):
    load_in_4bit = npu_guard_4bit(load_in_4bit, "discriminator(trainable)")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        num_labels=4,
        torch_dtype=choose_torch_dtype(bf16),
        quantization_config=maybe_build_bnb_config(load_in_4bit, bf16),
        device_map="auto" if load_in_4bit and not is_npu_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Prepare a quantized model for QLoRA training.
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    # Enable input gradients for gradient checkpointing.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    return model, tokenizer


def load_eval_discriminator(
    base_model_path: str,
    lora_path: Optional[str],
    bf16: bool,
    load_in_4bit: bool,
):
    load_in_4bit = npu_guard_4bit(load_in_4bit, "discriminator(eval)")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        num_labels=4,
        torch_dtype=choose_torch_dtype(bf16),
        quantization_config=maybe_build_bnb_config(load_in_4bit, bf16),
        device_map="auto" if load_in_4bit and not is_npu_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    model.eval()
    return model, tokenizer



# =========================================================
# Hugging Face batched generation helpers
# =========================================================

def build_sampling_prompt_and_framework(
    source_raw: Dict[str, Any],
    args,
    tokenizer,
) -> Tuple[str, str]:
    messages, framework = build_generator_messages_from_source(
        source_raw=source_raw,
        output_language_field=args.output_language_field,
        keep_extra_top_fields=args.keep_extra_top_fields,
        min_nodes=args.min_nodes,
        min_edges=args.min_edges,
    )
    prompt_text = render_generator_prompt(tokenizer, messages, enable_thinking=not args.disable_thinking)
    return prompt_text, framework











def _init_sampling_states_for_batch(
    batch_source_raws: List[Dict[str, Any]],
    args,
    tokenizer,
    batch_start_idx: int,
    total_samples: Optional[int],
) -> List[Dict[str, Any]]:
    preview_chars = getattr(args, "sampling_preview_chars", 120)
    states: List[Dict[str, Any]] = []

    for local_idx, source_raw in enumerate(batch_source_raws):
        global_idx = batch_start_idx + local_idx
        prompt_text, framework = build_sampling_prompt_and_framework(source_raw, args, tokenizer)
        source_id = source_raw.get("id", f"idx_{global_idx}")
        sample_pos = f"{global_idx + 1}/{total_samples}" if total_samples is not None else str(global_idx + 1)

        states.append({
            "source_raw": source_raw,
            "framework": framework,
            "prompt_text": prompt_text,
            "collected": [],
            "seen_inputs": set(),
            "sample_start_time": time.time(),
            "debug_info": {
                "sample_idx": global_idx,
                "sample_pos": sample_pos,
                "source_id": source_id,
                "framework": framework,
                "generator_sample_max_new_tokens": int(args.generator_sample_max_new_tokens),
                "generator_sample_retry_rounds": int(args.generator_sample_retry_rounds),
                "target_num_synthetics": 3,
                "num_generate_calls": 0,
                "num_total_sequences_requested": 0,
                "num_total_sequences_returned": 0,
                "num_parse_failed": 0,
                "num_empty_input": 0,
                "num_duplicate_input": 0,
                "num_valid_collected": 0,
                "accepted_samples": [],
                "round_logs": [],
                "status": "running",
                "prompt_preview": prompt_text[:preview_chars],
            },
        })
    return states


def _finalize_sampling_states(states: List[Dict[str, Any]], sampling_log_path: Optional[str]) -> List[Dict[str, Any]]:
    records = []
    for state in states:
        debug_info = state["debug_info"]
        total_elapsed = time.time() - state["sample_start_time"]
        debug_info["num_valid_collected"] = len(state["collected"])
        debug_info["elapsed_sec"] = round(total_elapsed, 4)
        debug_info["status"] = "ok" if len(state["collected"]) == 3 else "insufficient_valid_samples"
        records.append(debug_info)
    if sampling_log_path is not None and records:
        append_jsonl(sampling_log_path, records)
    return records


def _register_candidate(state: Dict[str, Any], parsed: Optional[Dict[str, Any]], preview_chars: int) -> None:
    debug_info = state["debug_info"]
    if parsed is None:
        debug_info["num_parse_failed"] += 1
        return

    input_text = parsed["input"].strip()
    if not input_text:
        debug_info["num_empty_input"] += 1
        return

    if input_text in state["seen_inputs"]:
        debug_info["num_duplicate_input"] += 1
        return

    state["seen_inputs"].add(input_text)
    state["collected"].append(parsed)
    debug_info["accepted_samples"].append({
        "accepted_index": len(state["collected"]),
        "input_length": len(input_text),
        "num_nodes": len(parsed.get("nodes", [])),
        "num_edges": len(parsed.get("edges", [])),
        "input_preview": input_text[:preview_chars],
    })


def sample_synthetics_for_real_batch_hf(
    model,
    tokenizer,
    batch_source_raws: List[Dict[str, Any]],
    args,
    device: str,
    batch_start_idx: int = 0,
    total_samples: Optional[int] = None,
    sampling_log_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    states = _init_sampling_states_for_batch(batch_source_raws, args, tokenizer, batch_start_idx, total_samples)
    preview_chars = getattr(args, "sampling_preview_chars", 120)
    batch_first = batch_start_idx + 1
    batch_last = batch_start_idx + len(batch_source_raws)
    total_text = total_samples if total_samples is not None else "?"

    log_info(
        f"[SamplingBatch {batch_first}-{batch_last}/{total_text}] start | "
        f"backend=hf_batched | batch_size={len(batch_source_raws)} | "
        f"retry_rounds={args.generator_sample_retry_rounds}"
    )

    for retry_round in range(args.generator_sample_retry_rounds):
        active_indices = [i for i, state in enumerate(states) if len(state["collected"]) < 3]
        if not active_indices:
            break

        prompts = [states[i]["prompt_text"] for i in active_indices]
        for i in active_indices:
            states[i]["debug_info"]["num_generate_calls"] += 1

        num_return_sequences = max(
            1,
            min(
                args.generator_sampling_round_candidates,
                max(3 - len(states[i]["collected"]) for i in active_indices),
            ),
        )

        for i in active_indices:
            states[i]["debug_info"]["num_total_sequences_requested"] += num_return_sequences

        log_info(
            f"[SamplingBatch {batch_first}-{batch_last}/{total_text}] "
            f"retry {retry_round + 1}/{args.generator_sample_retry_rounds} | "
            f"active_samples={len(active_indices)} | "
            f"round_candidates={num_return_sequences}"
        )

        round_start_time = time.time()
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        validate_generation_lengths(
            enc, tokenizer, model, args.generator_prompt_max_length,
            args.generator_sample_max_new_tokens, "HF sampling",
        )
        # generate() includes the entire padded input prefix for every sequence.
        prompt_width = int(enc["input_ids"].shape[1])
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        prompt_token_lengths = list(map(int, input_lens))
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=args.generator_sample_max_new_tokens,
                do_sample=True,
                temperature=args.generator_sample_temperature,
                top_p=args.generator_sample_top_p,
                repetition_penalty=args.generator_sample_repetition_penalty,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        outputs = outputs.reshape(len(active_indices), num_return_sequences, -1)
        round_elapsed = time.time() - round_start_time
        log_info(
            f"[SamplingBatch {batch_first}-{batch_last}/{total_text}] "
            f"retry {retry_round + 1} generation finished | "
            f"elapsed={round_elapsed:.2f}s"
        )

        for active_pos, state_idx in enumerate(active_indices):
            state = states[state_idx]
            state["debug_info"]["prompt_token_length"] = prompt_token_lengths[active_pos]
            state["debug_info"]["num_total_sequences_returned"] += num_return_sequences

            before_valid = len(state["collected"])
            before_parse_failed = state["debug_info"]["num_parse_failed"]
            before_empty = state["debug_info"]["num_empty_input"]
            before_dup = state["debug_info"]["num_duplicate_input"]

            for seq in outputs[active_pos]:
                if len(state["collected"]) >= 3:
                    break
                new_tokens = seq[prompt_width:]
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                parsed = parse_qwen_generated_json(raw_text, expected_framework=state["framework"])
                _register_candidate(state, parsed, preview_chars)

            state["debug_info"]["round_logs"].append({
                "retry_round": retry_round + 1,
                "requested_num_return_sequences": num_return_sequences,
                "returned_num_sequences": num_return_sequences,
                "valid_this_round": len(state["collected"]) - before_valid,
                "parse_failed_this_round": state["debug_info"]["num_parse_failed"] - before_parse_failed,
                "empty_input_this_round": state["debug_info"]["num_empty_input"] - before_empty,
                "duplicate_input_this_round": state["debug_info"]["num_duplicate_input"] - before_dup,
                "collected_after_round": len(state["collected"]),
                "round_elapsed_sec": round(round_elapsed, 4),
                "backend": "hf_batched",
            })

    ok_count = sum(1 for state in states if len(state["collected"]) == 3)
    log_info(
        f"[SamplingBatch {batch_first}-{batch_last}/{total_text}] done | "
        f"success={ok_count}/{len(states)}"
    )
    _finalize_sampling_states(states, sampling_log_path)
    return [{"synthetics": state["collected"], "debug_info": state["debug_info"]} for state in states]




# =========================================================
# Discriminator scorer
# =========================================================

class DiscriminatorScorer:
    def __init__(self, model, tokenizer, max_length: int, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = resolve_token_budget(tokenizer, model, max_length)

        if device is None:
            device = get_default_device()
        self.device = device

        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.model.eval()

    @torch.no_grad()
    def score_group(self, candidates: List[str]) -> Dict[str, Any]:
        prompt = build_discriminator_prompt(candidates)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )
        check_token_length(int(inputs["input_ids"].shape[1]), self.max_length,
                           "Discriminator reward group")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred_index = int(torch.argmax(probs).item())
        return {
            "logits": logits.detach().cpu().tolist(),
            "probs": probs.detach().cpu().tolist(),
            "pred_index": pred_index,
        }


# =========================================================
# GRPO dataset
# =========================================================

def build_grpo_dataset(real_records: List[Dict[str, Any]], args, generator_tokenizer) -> Dataset:
    rows = []
    for i, source_raw in enumerate(real_records):
        messages, framework = build_generator_messages_from_source(
            source_raw=source_raw,
            output_language_field=args.output_language_field,
            keep_extra_top_fields=args.keep_extra_top_fields,
            min_nodes=args.min_nodes,
            min_edges=args.min_edges,
        )
        prompt_text = render_generator_prompt(generator_tokenizer, messages, enable_thinking=not args.disable_thinking)

        rows.append({
            "sample_id": str(i),
            "prompt": prompt_text,
            "real_input": str(source_raw["input"]).strip(),
            "real_sample_for_disc": serialize_sample_for_discriminator(
                source_raw,
                output_language_field=args.output_language_field,
                keep_extra_top_fields=args.keep_extra_top_fields,
            ),
            "expected_framework": framework,
        })
    return Dataset.from_list(rows)


# =========================================================
# Generator reward based on complete samples
# =========================================================

class GroupDiscriminatorReward:
    def __init__(
        self,
        disc_scorer: DiscriminatorScorer,
        num_generations: int,
        reward_mode: str = "neg_log_p_real",
        invalid_completion_reward: float = -1.0,
        seed: int = 42,
        log_path: Optional[str] = None,
        output_language_field: bool = False,
        keep_extra_top_fields: bool = False,
        discriminator_reward_weight: float = 1.0,
        exact_match_reward_weight: float = 0.5,
    ):
        self.disc_scorer = disc_scorer
        self.num_generations = num_generations
        self.reward_mode = reward_mode
        self.invalid_completion_reward = invalid_completion_reward
        self.rng = random.Random(seed)
        self.log_path = log_path
        self.output_language_field = output_language_field
        self.keep_extra_top_fields = keep_extra_top_fields
        self.discriminator_reward_weight = discriminator_reward_weight
        self.exact_match_reward_weight = exact_match_reward_weight

    def _completion_disc_reward(self, real_sample_for_disc_text: str, synth_candidate_text: str) -> Dict[str, Any]:
        anchor_fake_texts = build_fixed_anchor_fake_texts(
            real_sample_for_disc_text,
            output_language_field=self.output_language_field,
            keep_extra_top_fields=self.keep_extra_top_fields,
        )
        candidates = [real_sample_for_disc_text, synth_candidate_text] + anchor_fake_texts

        perm = list(range(4))
        self.rng.shuffle(perm)
        shuffled_candidates = [candidates[p] for p in perm]
        true_pos = perm.index(0)

        disc_out = self.disc_scorer.score_group(shuffled_candidates)
        probs = disc_out["probs"]
        pred = disc_out["pred_index"]

        if self.reward_mode == "zero_one":
            disc_reward = 1.0 if pred != true_pos else -1.0
        else:
            p_real = max(float(probs[true_pos]), 1e-8)
            disc_reward = -math.log(p_real)

        return {
            "disc_reward": disc_reward,
            "shuffled_candidates": shuffled_candidates,
            "true_real_position": true_pos,
            "pred_index": pred,
            "probs": probs,
            "logits": disc_out["logits"],
        }

    def __call__(self, prompts, completions, real_input, real_sample_for_disc, expected_framework, sample_id, **kwargs):
        n = len(completions)
        rewards = [self.invalid_completion_reward for _ in range(n)]
        log_records = []

        buckets = {}
        for idx in range(n):
            sid = str(sample_id[idx])
            if sid not in buckets:
                buckets[sid] = {
                    "indices": [],
                    "real_input": real_input[idx],
                    "real_sample_for_disc": real_sample_for_disc[idx],
                    "expected_framework": expected_framework[idx],
                    "completions": [],
                }
            buckets[sid]["indices"].append(idx)
            buckets[sid]["completions"].append(completions[idx])

        for sid, group in buckets.items():
            idxs = group["indices"]
            real_text = group["real_input"]
            real_sample_for_disc_text = group["real_sample_for_disc"]
            expected_fw = group["expected_framework"]
            raw_completions = group["completions"]

            if len(raw_completions) != self.num_generations:
                for i in idxs:
                    rewards[i] = self.invalid_completion_reward
                log_records.append({
                    "sample_id": sid,
                    "expected_framework": expected_fw,
                    "real_input": real_text,
                    "real_sample_for_discriminator": real_sample_for_disc_text,
                    "completion_level_records": [],
                    "status": "invalid_group_size",
                })
                continue

            completion_records = []
            for local_idx, comp in enumerate(raw_completions):
                info = parse_qwen_generated_json_with_stats(comp, expected_framework=expected_fw)
                sample = info["sample"]

                if sample is None:
                    total_reward = self.invalid_completion_reward
                    rewards[idxs[local_idx]] = total_reward
                    completion_records.append({
                        "completion_index": local_idx,
                        "status": "invalid_parse",
                        "processed_generator_output": None,
                        "num_raw_nodes_with_text": info["num_raw_nodes_with_text"],
                        "num_exact_match_nodes": info["num_exact_match_nodes"],
                        "exact_match_ratio": info["exact_match_ratio"],
                        "exact_match_rule_reward": None,
                        "discriminator_reward": None,
                        "total_reward": total_reward,
                        "discriminator_processed_output": None,
                    })
                    continue

                synth_candidate_text = serialize_sample_for_discriminator(
                    sample,
                    output_language_field=self.output_language_field,
                    keep_extra_top_fields=self.keep_extra_top_fields,
                )
                disc_info = self._completion_disc_reward(real_sample_for_disc_text, synth_candidate_text)
                span_reward = exact_match_rule_reward(
                    info["exact_match_ratio"],
                    has_any_node_text=(info["num_raw_nodes_with_text"] > 0),
                )
                total_reward = (
                    self.discriminator_reward_weight * float(disc_info["disc_reward"])
                    + self.exact_match_reward_weight * float(span_reward)
                )
                rewards[idxs[local_idx]] = total_reward
                completion_records.append({
                    "completion_index": local_idx,
                    "status": "ok",
                    "processed_generator_output": sample,
                    "num_raw_nodes_with_text": info["num_raw_nodes_with_text"],
                    "num_exact_match_nodes": info["num_exact_match_nodes"],
                    "exact_match_ratio": info["exact_match_ratio"],
                    "exact_match_rule_reward": span_reward,
                    "discriminator_reward": disc_info["disc_reward"],
                    "total_reward": total_reward,
                    "discriminator_processed_output": {
                        "shuffled_candidates": disc_info["shuffled_candidates"],
                        "true_real_position": disc_info["true_real_position"],
                        "pred_index": disc_info["pred_index"],
                        "probs": disc_info["probs"],
                        "logits": disc_info["logits"],
                    },
                })

            log_records.append({
                "sample_id": sid,
                "expected_framework": expected_fw,
                "real_input": real_text,
                "real_sample_for_discriminator": real_sample_for_disc_text,
                "completion_level_records": completion_records,
                "status": "ok",
            })

        if self.log_path is not None and len(log_records) > 0:
            append_jsonl(self.log_path, log_records)

        return rewards


# =========================================================
# Generate incremental discriminator data with NPU support
# =========================================================

def sample_three_synthetics_for_one_real(
    model,
    tokenizer,
    source_raw: Dict[str, Any],
    args,
    device: str,
    sample_idx: Optional[int] = None,
    total_samples: Optional[int] = None,
    sampling_log_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    messages, framework = build_generator_messages_from_source(
        source_raw=source_raw,
        output_language_field=args.output_language_field,
        keep_extra_top_fields=args.keep_extra_top_fields,
        min_nodes=args.min_nodes,
        min_edges=args.min_edges,
    )
    prompt_text = render_generator_prompt(tokenizer, messages, enable_thinking=not args.disable_thinking)

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    validate_generation_lengths(
        inputs, tokenizer, model, args.generator_prompt_max_length,
        args.generator_sample_max_new_tokens, "HF single-sample generation",
    )
    prompt_len = inputs["input_ids"].shape[1]
    collected = []
    seen_inputs = set()

    preview_chars = getattr(args, "sampling_preview_chars", 120)
    source_id = source_raw.get("id", f"idx_{sample_idx if sample_idx is not None else 'unknown'}")
    sample_pos = f"{sample_idx + 1}/{total_samples}" if sample_idx is not None and total_samples is not None else "?"
    sample_start_time = time.time()

    debug_info = {
        "sample_idx": sample_idx,
        "sample_pos": sample_pos,
        "source_id": source_id,
        "framework": framework,
        "prompt_token_length": int(prompt_len),
        "generator_sample_max_new_tokens": int(args.generator_sample_max_new_tokens),
        "generator_sample_retry_rounds": int(args.generator_sample_retry_rounds),
        "target_num_synthetics": 3,
        "num_generate_calls": 0,
        "num_total_sequences_requested": 0,
        "num_total_sequences_returned": 0,
        "num_parse_failed": 0,
        "num_empty_input": 0,
        "num_duplicate_input": 0,
        "num_valid_collected": 0,
        "accepted_samples": [],
        "round_logs": [],
        "status": "running",
    }

    log_info(
        f"[Sampling {sample_pos}] start | source_id={source_id} | framework={framework} | "
        f"prompt_tokens={prompt_len} | max_new_tokens={args.generator_sample_max_new_tokens} | "
        f"retry_rounds={args.generator_sample_retry_rounds}"
    )

    for retry_round in range(args.generator_sample_retry_rounds):
        need = 3 - len(collected)
        if need <= 0:
            break

        log_info(f"[Sampling {sample_pos}] round {retry_round + 1} | need={need} | generating...")
        round_start_time = time.time()
        round_parse_failed = 0
        round_empty_input = 0
        round_duplicate_input = 0
        round_valid = 0

        debug_info["num_generate_calls"] += 1
        debug_info["num_total_sequences_requested"] += need

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.generator_sample_max_new_tokens,
                do_sample=True,
                temperature=args.generator_sample_temperature,
                top_p=args.generator_sample_top_p,
                repetition_penalty=args.generator_sample_repetition_penalty,
                num_return_sequences=need,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        debug_info["num_total_sequences_returned"] += len(outputs)

        gen_elapsed = time.time() - round_start_time
        log_info(
            f"[Sampling {sample_pos}] round {retry_round + 1} | generate took {gen_elapsed:.2f}s, returned {len(outputs)} sequences")

        for seq in outputs:
            new_tokens = seq[prompt_len:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            parsed = parse_qwen_generated_json(raw_text, expected_framework=framework)
            if parsed is None:
                round_parse_failed += 1
                debug_info["num_parse_failed"] += 1
                continue

            input_text = parsed["input"].strip()
            if not input_text:
                round_empty_input += 1
                debug_info["num_empty_input"] += 1
                continue

            if input_text in seen_inputs:
                round_duplicate_input += 1
                debug_info["num_duplicate_input"] += 1
                continue

            seen_inputs.add(input_text)
            collected.append(parsed)
            round_valid += 1

            debug_info["accepted_samples"].append({
                "accepted_index": len(collected),
                "input_length": len(input_text),
                "num_nodes": len(parsed.get("nodes", [])),
                "num_edges": len(parsed.get("edges", [])),
                "input_preview": input_text[:preview_chars],
            })

            if len(collected) >= 3:
                break

        round_elapsed = time.time() - round_start_time
        debug_info["round_logs"].append({
            "retry_round": retry_round + 1,
            "requested_num_return_sequences": need,
            "returned_num_sequences": len(outputs),
            "valid_this_round": round_valid,
            "parse_failed_this_round": round_parse_failed,
            "empty_input_this_round": round_empty_input,
            "duplicate_input_this_round": round_duplicate_input,
            "collected_after_round": len(collected),
            "round_elapsed_sec": round(round_elapsed, 4),
        })

        log_info(
            f"[Sampling {sample_pos}] retry {retry_round + 1}/{args.generator_sample_retry_rounds} | "
            f"requested={need} | returned={len(outputs)} | valid={round_valid} | "
            f"parse_failed={round_parse_failed} | empty={round_empty_input} | "
            f"duplicate={round_duplicate_input} | collected={len(collected)}/3 | "
            f"round_time={round_elapsed:.2f}s"
        )

    total_elapsed = time.time() - sample_start_time
    debug_info["num_valid_collected"] = len(collected)
    debug_info["elapsed_sec"] = round(total_elapsed, 4)
    debug_info["status"] = "ok" if len(collected) == 3 else "insufficient_valid_samples"

    log_info(
        f"[Sampling {sample_pos}] done | source_id={source_id} | status={debug_info['status']} | "
        f"collected={len(collected)}/3 | parse_failed={debug_info['num_parse_failed']} | "
        f"duplicate={debug_info['num_duplicate_input']} | elapsed={total_elapsed:.2f}s"
    )

    if sampling_log_path is not None:
        append_jsonl(sampling_log_path, [debug_info])

    return collected, debug_info


def build_disc_examples_from_generated_groups(
    generated_groups: List[Dict[str, Any]],
    num_permutations_per_group: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    examples = []

    for g in generated_groups:
        base = [
            {"text": g["real_sample_for_disc"], "is_real": 1},
            {"text": g["synthetic_sample_texts"][0], "is_real": 0},
            {"text": g["synthetic_sample_texts"][1], "is_real": 0},
            {"text": g["synthetic_sample_texts"][2], "is_real": 0},
        ]

        seen = set()
        trials = 0
        while len(seen) < num_permutations_per_group and trials < 100:
            trials += 1
            perm = list(range(4))
            rng.shuffle(perm)
            tup = tuple(perm)
            if tup in seen:
                continue
            seen.add(tup)

            shuffled = [base[i] for i in perm]
            cands = [x["text"] for x in shuffled]
            label = next(i for i, x in enumerate(shuffled) if x["is_real"] == 1)

            examples.append({
                "group_id": g["sample_id"],
                "prompt": build_discriminator_prompt(cands),
                "label": label,
                "real_position": label,
            })

    return examples


def split_disc_examples_by_group(
    examples: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(examples) == 0:
        return [], []

    group_ids = sorted(list(set(ex["group_id"] for ex in examples)))
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    if len(group_ids) == 1:
        return examples, []

    val_size = max(1, int(len(group_ids) * val_ratio))
    val_size = min(val_size, len(group_ids) - 1)
    val_groups = set(group_ids[:val_size])

    train_data, val_data = [], []
    for ex in examples:
        if ex["group_id"] in val_groups:
            val_data.append(ex)
        else:
            train_data.append(ex)
    return train_data, val_data


def get_external_fixed_val_examples_path(args) -> str:
    return os.path.join(args.output_dir, "disc_external_val_fixed_examples.jsonl")


def load_external_fixed_val_examples_if_available(args) -> Optional[List[Dict[str, Any]]]:
    if not getattr(args, "disc_external_val_fixed_candidates", False):
        return None
    fixed_path = get_external_fixed_val_examples_path(args)
    if os.path.exists(fixed_path):
        examples = load_jsonl(fixed_path)
        log_info(f"Loaded fixed external validation candidates from: {fixed_path} | examples={len(examples)}")
        return examples
    return None


def maybe_build_or_load_external_val_examples(
    args,
    round_idx: int,
    round_out: str,
    external_val_records: List[Dict[str, Any]],
    gen_tokenizer,
    gen_model=None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if external_val_records is None or len(external_val_records) == 0:
        raise ValueError("disc_val_ratio=0 requires a nonempty external validation dataset of real samples (--disc_external_val_file).")

    fixed_examples_path = get_external_fixed_val_examples_path(args)

    if getattr(args, "disc_external_val_fixed_candidates", False) and os.path.exists(fixed_examples_path):
        val_examples = load_jsonl(fixed_examples_path)
        log_info(
            f"[Round {round_idx}] using cached fixed external validation candidates | "
            f"path={fixed_examples_path} | val_examples={len(val_examples)}"
        )
        return val_examples

    log_info(f"[Round {round_idx}] disc_val_ratio=0, using external validation real data: {len(external_val_records)} records")
    val_generated_groups, val_failed_count, val_sampling_elapsed = generate_synthetic_groups_for_records(
        records=external_val_records,
        args=args,
        round_idx=round_idx,
        round_out=round_out,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
        device=device,
        stage_tag="val",
        sample_id_prefix="val_",
    )
    log_info(
        f"[Round {round_idx}] external validation sampling summary | "
        f"valid_groups={len(val_generated_groups)} | failed_groups={val_failed_count} | "
        f"elapsed={val_sampling_elapsed / 60:.2f} min"
    )

    val_examples = build_disc_examples_from_generated_groups(
        val_generated_groups,
        num_permutations_per_group=args.disc_num_permutations_per_group,
        seed=args.seed + 3000,
    ) if len(val_generated_groups) > 0 else []

    if len(val_examples) > 0:
        save_jsonl(os.path.join(round_out, "disc_val_pool_preview.jsonl"), val_examples[:300])

    if getattr(args, "disc_external_val_fixed_candidates", False):
        save_jsonl(fixed_examples_path, val_examples)
        log_info(
            f"[Round {round_idx}] saved fixed external validation candidates | "
            f"path={fixed_examples_path} | val_examples={len(val_examples)}"
        )

    return val_examples


def generate_synthetic_groups_for_records(
    records: List[Dict[str, Any]],
    args,
    round_idx: int,
    round_out: str,
    gen_tokenizer,
    gen_model=None,
    device: Optional[str] = None,
    stage_tag: str = "train",
    sample_id_prefix: str = "",
) -> Tuple[List[Dict[str, Any]], int, float]:
    generated_groups = []
    failed_count = 0

    sampling_log_path = os.path.join(round_out, f"{stage_tag}_sampling_stage_logs.jsonl")
    generated_groups_path = os.path.join(round_out, f"{stage_tag}_generated_groups.jsonl")





    if os.path.exists(generated_groups_path):    # Reuse an existing sampling artifact.
        log_info(f"[Resume] Found existing generated groups at {generated_groups_path}; loading them and skipping sampling.")
        with open(generated_groups_path, "r", encoding="utf-8") as f:
            generated_groups = [json.loads(line) for line in f]
        # Return cached groups without performing sampling.
        return generated_groups, 0, 0.0




    if os.path.exists(sampling_log_path):
        os.remove(sampling_log_path)

    sampling_stage_start = time.time()
    log_info(
        f"[Round {round_idx}] start {stage_tag} sampling for discriminator | "
        f"total_real_records={len(records)} | target_per_real=3 | "
        f"max_new_tokens={args.generator_sample_max_new_tokens} | "
        f"retry_rounds={args.generator_sample_retry_rounds}"
    )

    num_batches = math.ceil(len(records) / args.generator_sampling_batch_size) if len(records) > 0 else 0
    for batch_id, batch_start in enumerate(range(0, len(records), args.generator_sampling_batch_size), start=1):
        batch_records = records[batch_start: batch_start + args.generator_sampling_batch_size]
        batch_end = batch_start + len(batch_records)

        log_info(
            f"[Round {round_idx}] {stage_tag} sampling batch {batch_id}/{num_batches} start | "
            f"indices={batch_start + 1}-{batch_end} | backend=hf_batched"
        )

        batch_results = sample_synthetics_for_real_batch_hf(
            model=gen_model,
            tokenizer=gen_tokenizer,
            batch_source_raws=batch_records,
            args=args,
            device=device,
            batch_start_idx=batch_start,
            total_samples=len(records),
            sampling_log_path=sampling_log_path,
        )

        for local_idx, result in enumerate(batch_results):
            global_idx = batch_start + local_idx
            source_raw = batch_records[local_idx]
            synths = result["synthetics"]

            if len(synths) < 3:
                failed_count += 1
            else:
                sample_id = f"{sample_id_prefix}{global_idx}" if sample_id_prefix else str(global_idx)
                generated_groups.append({
                    "sample_id": sample_id,
                    "real_input": str(source_raw["input"]).strip(),
                    "real_sample_for_disc": serialize_sample_for_discriminator(
                        source_raw,
                        output_language_field=args.output_language_field,
                        keep_extra_top_fields=args.keep_extra_top_fields,
                    ),
                    "synthetic_samples": synths,
                    "synthetic_sample_texts": [
                        serialize_sample_for_discriminator(
                            s,
                            output_language_field=args.output_language_field,
                            keep_extra_top_fields=args.keep_extra_top_fields,
                        )
                        for s in synths
                    ],
                })

        batch_success = sum(1 for r in batch_results if len(r["synthetics"]) >= 3)
        batch_failed = len(batch_results) - batch_success
        log_info(
            f"[Round {round_idx}] {stage_tag} sampling batch {batch_id}/{num_batches} finished | "
            f"batch_success={batch_success} | batch_failed={batch_failed} | "
            f"total_valid_groups={len(generated_groups)} | total_failed_groups={failed_count}"
        )

        processed = min(batch_start + len(batch_records), len(records))
        if processed % args.sampling_progress_log_interval == 0 or processed == len(records):
            elapsed = time.time() - sampling_stage_start
            avg_time = elapsed / processed
            log_info(
                f"[Round {round_idx}] {stage_tag} sampling progress | "
                f"{processed}/{len(records)} samples processed | "
                f"valid_groups={len(generated_groups)} | failed_groups={failed_count} | "
                f"avg_time_per_sample={avg_time:.2f}s | elapsed={elapsed / 60:.2f} min"
            )

        del batch_results
        gc.collect()
        clear_accelerator_cache()

    save_jsonl(generated_groups_path, generated_groups)

    sampling_stage_elapsed = time.time() - sampling_stage_start
    log_info(
        f"[Round {round_idx}] {stage_tag} sampling finished | valid generated groups={len(generated_groups)} | "
        f"failed groups={failed_count} | elapsed={sampling_stage_elapsed / 60:.2f} min | "
        f"sampling_log={sampling_log_path}"
    )
    return generated_groups, failed_count, sampling_stage_elapsed


# =========================================================
# replay buffer
# =========================================================

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def add(self, items: List[Dict[str, Any]]):
        self.buffer.extend(copy.deepcopy(items))
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, n: int, seed: int) -> List[Dict[str, Any]]:
        if n <= 0 or len(self.buffer) == 0:
            return []
        if n >= len(self.buffer):
            return copy.deepcopy(self.buffer)
        rng = random.Random(seed)
        return copy.deepcopy(rng.sample(self.buffer, n))

    def __len__(self):
        return len(self.buffer)


# =========================================================
# Discriminator training dataset
# =========================================================

class DiscriminatorTrainDataset(TorchDataset):
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


def compute_disc_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


@torch.no_grad()
def collect_discriminator_processed_outputs(
    examples: List[Dict[str, Any]],
    disc_scorer: DiscriminatorScorer,
) -> List[Dict[str, Any]]:
    records = []

    for ex in examples:
        prompt = ex["prompt"]
        inputs = disc_scorer.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )
        check_token_length(int(inputs["input_ids"].shape[1]), disc_scorer.max_length,
                           "Discriminator diagnostic group")
        inputs = {k: v.to(disc_scorer.device) for k, v in inputs.items()}
        outputs = disc_scorer.model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred_index = int(torch.argmax(probs).item())

        records.append({
            "group_id": ex.get("group_id"),
            "label": ex.get("label"),
            "real_position": ex.get("real_position"),
            "pred_index": pred_index,
            "correct": bool(pred_index == ex.get("label")),
            "probs": probs.detach().cpu().tolist(),
            "logits": logits.detach().cpu().tolist(),
            "prompt": prompt,
        })

    return records


# =========================================================
# One generator GRPO update round
# =========================================================

def run_one_generator_grpo_round(
    real_records: List[Dict[str, Any]],
    args,
    current_generator_lora_path: Optional[str],
    current_discriminator_lora_path: Optional[str],
    round_idx: int,
) -> str:
    round_out = os.path.join(args.output_dir, f"generator_round_{round_idx}")
    ensure_dir(round_out)

    if getattr(args, "skip_completed_rounds", True) and is_complete_lora_adapter_dir(round_out):
        log_info(f"[Round {round_idx}] completed generator adapter found, skip GRPO training: {round_out}")
        return round_out

    resume_checkpoint = resolve_grpo_resume_checkpoint(args, round_out)
    if resume_checkpoint is not None:
        log_info(f"[Round {round_idx}] auto-resume GRPO from checkpoint: {resume_checkpoint}")
    elif args.grpo_save_steps <= 0:
        log_info(f"[Round {round_idx}] GRPO checkpoint saving is disabled because grpo_save_steps <= 0.")

    gen_model, gen_tokenizer = load_trainable_generator(
        base_model_path=args.generator_base_model_path,
        lora_path=current_generator_lora_path,
        bf16=args.bf16,
        load_in_4bit=args.generator_load_in_4bit,
    )

    disc_model, disc_tokenizer = load_eval_discriminator(
        base_model_path=args.discriminator_base_model_path,
        lora_path=current_discriminator_lora_path,
        bf16=args.bf16,
        load_in_4bit=args.discriminator_load_in_4bit,
    )
    disc_scorer = DiscriminatorScorer(
        disc_model,
        disc_tokenizer,
        max_length=args.discriminator_max_length,
        device=args.discriminator_device,
    )

    grpo_dataset = build_grpo_dataset(real_records, args, gen_tokenizer)
    prompt_budget = resolve_token_budget(gen_tokenizer, gen_model, args.generator_prompt_max_length)
    context_budget = resolve_token_budget(
        gen_tokenizer, gen_model, prompt_budget + args.grpo_max_completion_length,
    )
    for index, example in enumerate(grpo_dataset):
        # GRPO tokenizes these already-rendered prompts without extra special tokens.
        length = len(gen_tokenizer(example["prompt"], add_special_tokens=False,
                                   truncation=False)["input_ids"])
        check_token_length(length, prompt_budget, f"GRPO prompt {index}")
        check_token_length(length + args.grpo_max_completion_length, context_budget,
                           f"GRPO prompt {index} plus reserved completion")

    generator_reward_log_path = os.path.join(round_out, "generator_discriminator_processed_outputs.jsonl")
    if os.path.exists(generator_reward_log_path):
        if resume_checkpoint is None:
            os.remove(generator_reward_log_path)
            log_info(f"[Round {round_idx}] existing GRPO reward log removed for a fresh run: {generator_reward_log_path}")
        else:
            log_info(f"[Round {round_idx}] keep existing GRPO reward log and append after resume: {generator_reward_log_path}")

    save_json_atomic(os.path.join(round_out, "grpo_round_state.json"), {
        "round_idx": round_idx,
        "status": "running",
        "round_out": round_out,
        "resume_checkpoint": resume_checkpoint,
        "grpo_save_steps": args.grpo_save_steps,
        "grpo_save_total_limit": args.grpo_save_total_limit,
        "save_only_model": args.grpo_save_only_model,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    reward_func = GroupDiscriminatorReward(
        disc_scorer=disc_scorer,
        num_generations=args.grpo_num_generations,
        reward_mode=args.reward_mode,
        invalid_completion_reward=args.invalid_completion_reward,
        seed=args.seed + round_idx,
        log_path=generator_reward_log_path,
        output_language_field=args.output_language_field,
        keep_extra_top_fields=args.keep_extra_top_fields,
        discriminator_reward_weight=args.grpo_discriminator_reward_weight,
        exact_match_reward_weight=args.grpo_exact_match_reward_weight,
    )

    grpo_kwargs = dict(
        output_dir=round_out,
        per_device_train_batch_size=args.grpo_per_device_train_batch_size,
        gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
        learning_rate=args.grpo_learning_rate,
        num_train_epochs=args.grpo_num_train_epochs,
        max_steps=args.grpo_max_steps,
        logging_steps=args.grpo_logging_steps,
        save_steps=args.grpo_save_steps,
        save_strategy="steps" if args.grpo_save_steps > 0 else "no",
        save_total_limit=args.grpo_save_total_limit,
        report_to="none",
        bf16=args.bf16,
        fp16=(not args.bf16 and is_cuda_available()),
        remove_unused_columns=False,
        num_generations=args.grpo_num_generations,
        max_completion_length=args.grpo_max_completion_length,
        temperature=args.grpo_temperature,
        top_p=args.grpo_top_p,
        repetition_penalty=args.grpo_repetition_penalty,
        beta=args.grpo_beta,
        epsilon=args.grpo_epsilon,
        num_iterations=args.grpo_num_iterations,
        scale_rewards=args.grpo_scale_rewards,
        loss_type=args.grpo_loss_type,
        # For true resume, this must stay False so optimizer/scheduler/rng states are saved.
        save_only_model=args.grpo_save_only_model,
        # generation_kwargs={
        #     "use_cache": True,
        #     "pad_token_id": gen_tokenizer.pad_token_id,
        #     "eos_token_id": gen_tokenizer.eos_token_id,
        # },
    )
    grpo_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=gen_model,
        args=grpo_args,
        reward_funcs=reward_func.__call__,
        train_dataset=grpo_dataset,
        processing_class=gen_tokenizer,
    )
    log_info(f"[Round {round_idx}] GRPO trainer initialization complete. Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        log_info(f"[Round {round_idx}] GRPO interrupted by KeyboardInterrupt. Latest periodic checkpoint remains under: {round_out}")
        save_json_atomic(os.path.join(round_out, "grpo_round_state.json"), {
            "round_idx": round_idx,
            "status": "interrupted",
            "round_out": round_out,
            "latest_checkpoint": find_latest_trainer_checkpoint(round_out),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise
    log_info(f"[Round {round_idx}] GRPO training finished.")

    trainer.model.save_pretrained(round_out)
    gen_tokenizer.save_pretrained(round_out)
    save_json_atomic(os.path.join(round_out, "grpo_round_state.json"), {
        "round_idx": round_idx,
        "status": "complete",
        "round_out": round_out,
        "latest_checkpoint": find_latest_trainer_checkpoint(round_out),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    del trainer
    del gen_model
    del gen_tokenizer
    del disc_scorer
    del disc_model
    del disc_tokenizer
    gc.collect()
    clear_accelerator_cache()

    return round_out


# =========================================================
# One incremental discriminator training round
# =========================================================

def run_one_discriminator_incremental_round(
    real_records: List[Dict[str, Any]],
    args,
    current_generator_lora_path: Optional[str],
    current_discriminator_lora_path: Optional[str],
    replay_buffer: ReplayBuffer,
    round_idx: int,
    external_val_records: Optional[List[Dict[str, Any]]] = None,
) -> str:
    round_out = os.path.join(args.output_dir, f"discriminator_round_{round_idx}")
    ensure_dir(round_out)

    if getattr(args, "skip_completed_rounds", True) and is_complete_lora_adapter_dir(round_out):
        log_info(f"[Round {round_idx}] completed discriminator adapter found, skip discriminator training: {round_out}")
        return round_out

    # Stage 1: sample synthetic data with the current generator.
    gen_model = None
    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.generator_base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"
    device = None

    gen_model, _ = load_eval_generator(
        base_model_path=args.generator_base_model_path,
        lora_path=current_generator_lora_path,
        bf16=args.bf16,
        load_in_4bit=args.generator_load_in_4bit,
    )
    device = get_runtime_device(args.generator_device)
    try:
        gen_model.to(device)
    except Exception:
        pass
    gen_model.eval()
    log_info(f"[Round {round_idx}] incremental sampling backend=hf_batched | batch_size={args.generator_sampling_batch_size}")

    generated_groups, failed_count, sampling_stage_elapsed = generate_synthetic_groups_for_records(
        records=real_records,
        args=args,
        round_idx=round_idx,
        round_out=round_out,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
        device=device,
        stage_tag="train",
        sample_id_prefix="",
    )
    print(f"[Round {round_idx}] valid generated groups: {len(generated_groups)}, failed groups: {failed_count}")

    if len(generated_groups) == 0:
        print(f"[Round {round_idx}] no valid generated groups, skip discriminator update.")
        del gen_model
        del gen_tokenizer
        del device


        gc.collect()
        synchronize_accelerator()
        clear_accelerator_cache()
        gc.collect()
        clear_accelerator_cache()
        return current_discriminator_lora_path

    cached_fixed_val_examples = None
    if args.disc_val_ratio == 0 and args.disc_external_val_fixed_candidates:
        cached_fixed_val_examples = load_external_fixed_val_examples_if_available(args)

    val_examples = None
    if args.disc_val_ratio == 0:
        if cached_fixed_val_examples is not None:
            val_examples = cached_fixed_val_examples
        else:
            val_examples = maybe_build_or_load_external_val_examples(
                args=args,
                round_idx=round_idx,
                round_out=round_out,
                external_val_records=external_val_records,
                gen_tokenizer=gen_tokenizer,
                gen_model=gen_model,
                device=device,
            )

    # Stage 2: release generator memory after sampling.
    log_info(f"[Round {round_idx}] start aggressive generator memory cleanup before discriminator training")
    synchronize_accelerator()

    del gen_model
    del gen_tokenizer
    del device


    gc.collect()
    clear_accelerator_cache()
    collect_accelerator_ipc()
    synchronize_accelerator()
    gc.collect()
    clear_accelerator_cache()
    log_info(f"[Round {round_idx}] generator memory cleanup finished")

    # Stage 3: build the incremental discriminator training pool.
    log_info(f"[Round {round_idx}] building discriminator training pool ...")
    new_examples = build_disc_examples_from_generated_groups(
        generated_groups,
        num_permutations_per_group=args.disc_num_permutations_per_group,
        seed=args.seed + round_idx,
    )

    n_new = len(new_examples)
    n_old = int(n_new * args.disc_replay_ratio)
    old_examples = replay_buffer.sample(n_old, seed=args.seed + 1000 + round_idx)
    replay_buffer.add(new_examples)

    mixed_examples = copy.deepcopy(new_examples) + old_examples
    save_jsonl(os.path.join(round_out, "disc_train_pool_preview.jsonl"), mixed_examples[:300])

    del generated_groups
    del new_examples
    del old_examples
    gc.collect()
    clear_accelerator_cache()

    if args.disc_val_ratio == 0:
        train_examples = mixed_examples
        val_examples = val_examples if val_examples is not None else []
        if len(val_examples) > 0 and not os.path.exists(os.path.join(round_out, "disc_val_pool_preview.jsonl")):
            save_jsonl(os.path.join(round_out, "disc_val_pool_preview.jsonl"), val_examples[:300])
        log_info(
            f"[Round {round_idx}] validation source=external_real_file"
            f"{'_fixed_candidates' if args.disc_external_val_fixed_candidates else '_online_eval'} | "
            f"train_examples={len(train_examples)} | val_examples={len(val_examples)}"
        )
    else:
        train_examples, val_examples = split_disc_examples_by_group(
            mixed_examples,
            val_ratio=args.disc_val_ratio,
            seed=args.seed + 2000 + round_idx,
        )
        log_info(
            f"[Round {round_idx}] validation source=train_pool_split | "
            f"disc_val_ratio={args.disc_val_ratio} | train_examples={len(train_examples)} | val_examples={len(val_examples)}"
        )

    if len(train_examples) == 0:
        print(f"[Round {round_idx}] empty train_examples, skip discriminator update.")
        del mixed_examples
        gc.collect()
        clear_accelerator_cache()
        return current_discriminator_lora_path

    # Stage 4: load and train the discriminator.
    disc_model, disc_tokenizer = load_trainable_discriminator(
        base_model_path=args.discriminator_base_model_path,
        lora_path=current_discriminator_lora_path,
        bf16=args.bf16,
        load_in_4bit=args.discriminator_load_in_4bit,
    )

    args.discriminator_max_length = resolve_token_budget(
        disc_tokenizer, disc_model, args.discriminator_max_length,
    )
    validate_text_lengths(train_examples, disc_tokenizer, args.discriminator_max_length,
                          "Incremental discriminator train")
    validate_text_lengths(val_examples, disc_tokenizer, args.discriminator_max_length,
                          "Incremental discriminator validation")
    train_dataset = DiscriminatorTrainDataset(
        train_examples, disc_tokenizer, args.discriminator_max_length
    )
    val_dataset = DiscriminatorTrainDataset(
        val_examples, disc_tokenizer, args.discriminator_max_length
    ) if len(val_examples) > 0 else None

    collator = DataCollatorWithPadding(tokenizer=disc_tokenizer)

    training_args = TrainingArguments(
        output_dir=round_out,
        # overwrite_output_dir=True,
        per_device_train_batch_size=args.disc_per_device_train_batch_size,
        per_device_eval_batch_size=args.disc_per_device_eval_batch_size,
        gradient_accumulation_steps=args.disc_gradient_accumulation_steps,
        learning_rate=args.disc_learning_rate,
        weight_decay=args.disc_weight_decay,
        num_train_epochs=args.disc_num_train_epochs,
        max_steps=args.disc_max_steps,
        logging_steps=args.disc_logging_steps,
        eval_strategy="steps" if (args.disc_eval_steps > 0 and val_dataset is not None) else ("epoch" if val_dataset is not None else "no"),
        eval_steps=args.disc_eval_steps if args.disc_eval_steps > 0 else None,
        save_strategy="steps" if args.disc_save_steps > 0 else "epoch",
        save_steps=args.disc_save_steps if args.disc_save_steps > 0 else None,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        bf16=args.bf16,
        fp16=(not args.bf16 and is_cuda_available()),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    if hasattr(disc_model, "gradient_checkpointing_enable"):
        disc_model.gradient_checkpointing_enable()

    log_info(
        f"[Round {round_idx}] start discriminator training | "
        f"train_examples={len(train_examples)} | val_examples={len(val_examples)}"
    )

    trainer = Trainer(
        model=disc_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=disc_tokenizer,
        data_collator=collator,
        compute_metrics=compute_disc_metrics if val_dataset is not None else None,
    )
    trainer.train()
    log_info(f"[Round {round_idx}] discriminator training finished.")

    trainer.model.save_pretrained(round_out)
    disc_tokenizer.save_pretrained(round_out)

    discriminator_processed_outputs = collect_discriminator_processed_outputs(
        mixed_examples,
        DiscriminatorScorer(
            trainer.model,
            disc_tokenizer,
            max_length=args.discriminator_max_length,
            device=args.discriminator_device,
        ),
    )
    save_jsonl(
        os.path.join(round_out, "discriminator_processed_outputs.jsonl"),
        discriminator_processed_outputs,
    )

    # Stage 5: clean up after training.
    del trainer
    del disc_model
    del disc_tokenizer
    del train_dataset
    del val_dataset
    del collator
    del train_examples
    del val_examples
    del mixed_examples
    del discriminator_processed_outputs
    gc.collect()
    synchronize_accelerator()
    clear_accelerator_cache()
    collect_accelerator_ipc()
    gc.collect()
    clear_accelerator_cache()

    return round_out


# =========================================================
# Arguments
# =========================================================


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


def validate_generation_lengths(enc, tokenizer, model, prompt_budget: int,
                                max_new_tokens: int, context: str) -> None:
    """Reserve completion space after the padded prompt used by generate()."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    prompt_width = int(enc["input_ids"].shape[1])
    check_token_length(prompt_width, prompt_budget, f"{context} prompt")
    context_budget = resolve_token_budget(tokenizer, model, prompt_width + max_new_tokens)
    check_token_length(prompt_width + max_new_tokens, context_budget,
                       f"{context} prompt plus reserved completion")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--disable_internal_tee", action="store_true")
    parser.add_argument("--min_nodes", type=int, default=7)
    parser.add_argument("--min_edges", type=int, default=3)

    # Data
    parser.add_argument("--real_file", type=str, required=True)
    parser.add_argument("--disc_external_val_file", type=str, default=None)
    parser.add_argument("--disc_external_val_fixed_candidates", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)

    # Sampling logs
    parser.add_argument("--sampling_progress_log_interval", type=int, default=10)
    parser.add_argument("--sampling_preview_chars", type=int, default=120)

    # Model paths
    parser.add_argument("--generator_base_model_path", type=str, required=True)
    parser.add_argument("--generator_init_lora_path", type=str, required=True)
    parser.add_argument("--discriminator_base_model_path", type=str, required=True)
    parser.add_argument("--discriminator_init_lora_path", type=str, required=True)

    # Devices and precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--generator_load_in_4bit", action="store_true")
    parser.add_argument("--discriminator_load_in_4bit", action="store_true")
    parser.add_argument("--generator_device", type=str, default=None)
    parser.add_argument("--discriminator_device", type=str, default=None)

    # Generator prompts and output fields
    parser.add_argument("--output_language_field", action="store_true")
    parser.add_argument("--keep_extra_top_fields", action="store_true")

    # Outer training rounds
    parser.add_argument("--num_outer_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # ---------- GRPO ----------
    parser.add_argument("--grpo_num_generations", type=int, default=3)
    parser.add_argument("--grpo_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--grpo_learning_rate", type=float, default=1e-6)
    parser.add_argument("--grpo_num_train_epochs", type=float, default=1.0)
    parser.add_argument("--grpo_max_steps", type=int, default=-1)
    parser.add_argument("--grpo_logging_steps", type=int, default=10)
    parser.add_argument("--grpo_save_steps", type=int, default=50)
    parser.add_argument("--grpo_save_total_limit", type=int, default=3)
    parser.add_argument("--grpo_resume_from_checkpoint", type=str, default="auto", help="auto / none / explicit checkpoint path")
    parser.add_argument("--grpo_save_only_model", action="store_true", help="Only save model weights in checkpoints; this disables exact optimizer-state resume.")
    parser.add_argument("--grpo_max_completion_length", type=int, default=16384)
    parser.add_argument("--grpo_temperature", type=float, default=1.0)
    parser.add_argument("--grpo_top_p", type=float, default=1.0)
    parser.add_argument("--grpo_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--grpo_beta", type=float, default=0.0)
    parser.add_argument("--grpo_epsilon", type=float, default=0.2)
    parser.add_argument("--grpo_num_iterations", type=int, default=1)
    parser.add_argument("--grpo_scale_rewards", type=str, default="batch")
    parser.add_argument("--grpo_loss_type", type=str, default="dapo")

    # ---------- reward ----------
    parser.add_argument("--reward_mode", type=str, default="neg_log_p_real", choices=["neg_log_p_real", "zero_one"])
    parser.add_argument("--invalid_completion_reward", type=float, default=-1.0)
    parser.add_argument("--grpo_discriminator_reward_weight", type=float, default=1.0)
    parser.add_argument("--grpo_exact_match_reward_weight", type=float, default=0.5)

    # Incremental discriminator training
    parser.add_argument("--discriminator_max_length", type=int, default=32768)
    parser.add_argument("--disc_num_permutations_per_group", type=int, default=2)
    parser.add_argument("--disc_replay_ratio", type=float, default=0.5)
    parser.add_argument("--disc_replay_buffer_max_size", type=int, default=50000)
    # Positive ratio: split the generated training pool. Zero: use an external file.
    parser.add_argument("--disc_val_ratio", type=float, default=0.1,
                        help="Internal validation fraction; set to 0 to use --disc_external_val_file.")
    parser.add_argument("--disc_per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--disc_per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--disc_gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--disc_learning_rate", type=float, default=1e-5)
    parser.add_argument("--disc_weight_decay", type=float, default=0.01)
    parser.add_argument("--disc_num_train_epochs", type=float, default=20)
    parser.add_argument("--disc_max_steps", type=int, default=-1)
    parser.add_argument("--disc_logging_steps", type=int, default=10)
    parser.add_argument("--disc_eval_steps", type=int, default=0)
    parser.add_argument("--disc_save_steps", type=int, default=0)

    # Generator sampling
    parser.add_argument("--generator_prompt_max_length", type=int, default=16384)
    parser.add_argument("--generator_sample_max_new_tokens", type=int, default=16384)
    parser.add_argument("--generator_sample_temperature", type=float, default=1.0)
    parser.add_argument("--generator_sample_top_p", type=float, default=0.95)
    parser.add_argument("--generator_sample_repetition_penalty", type=float, default=1.05)
    parser.add_argument("--generator_sample_retry_rounds", type=int, default=4)
    parser.add_argument("--generator_sampling_batch_size", type=int, default=4)
    parser.add_argument("--generator_sampling_round_candidates", type=int, default=3)



    parser.add_argument("--no_skip_completed_rounds", dest="skip_completed_rounds", action="store_false",
                        help="Do not skip completed generator/discriminator round directories when resuming an output_dir.")
    parser.set_defaults(skip_completed_rounds=True)

    return parser.parse_args()


# =========================================================
# Main training loop
# =========================================================

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    if should_disable_internal_tee(args):
        log_path = os.path.join(args.output_dir, "train_terminal.log")
        log_f = None
        print(f"[Info] Internal tee disabled; outer launcher will capture stdout/stderr. target_log_file={log_path}", flush=True)
    else:
        log_path, log_f = init_log_file(args.output_dir)
        log_info(f"Training terminal log will also be saved to: {log_path}")
    set_seed(args.seed)

    real_records = load_jsonl(args.real_file)
    print(f"[Info] loaded real records: {len(real_records)}")
    log_info(
        f"GRPO reward config | discriminator_weight={args.grpo_discriminator_reward_weight} | "
        f"exact_match_weight={args.grpo_exact_match_reward_weight} | invalid_reward={args.invalid_completion_reward}"
    )

    external_val_records = None
    if args.disc_val_ratio == 0:
        if args.disc_external_val_file is None:
            raise ValueError("disc_val_ratio=0 requires --disc_external_val_file.")
        external_val_records = load_jsonl(args.disc_external_val_file)
        print(f"[Info] loaded external validation real records: {len(external_val_records)}")
    elif args.disc_external_val_file is not None:
        log_info("disc_val_ratio != 0, so --disc_external_val_file will be ignored and validation will be split from training pool.")

    if args.disc_val_ratio == 0 and args.disc_external_val_fixed_candidates:
        fixed_path = get_external_fixed_val_examples_path(args)
        if os.path.exists(fixed_path):
            log_info(f"Found existing fixed external validation candidates cache: {fixed_path}")
        else:
            log_info(f"Fixed external validation candidates will be created on first use and saved to: {fixed_path}")

    current_generator_lora_path = args.generator_init_lora_path
    current_discriminator_lora_path = args.discriminator_init_lora_path

    replay_buffer = ReplayBuffer(max_size=args.disc_replay_buffer_max_size)
    load_replay_buffer_state_if_available(args, replay_buffer)

    for round_idx in range(1, args.num_outer_rounds + 1):
        print(f"\n========== Outer Round {round_idx} / {args.num_outer_rounds} ==========")

        print("[Round] GRPO training generator ...")
        current_generator_lora_path = run_one_generator_grpo_round(
            real_records=real_records,
            args=args,
            current_generator_lora_path=current_generator_lora_path,
            current_discriminator_lora_path=current_discriminator_lora_path,
            round_idx=round_idx,
        )
        print(f"[Round] generator updated adapter saved to: {current_generator_lora_path}")

        print("[Round] incremental training discriminator ...")
        current_discriminator_lora_path = run_one_discriminator_incremental_round(
            real_records=real_records,
            args=args,
            current_generator_lora_path=current_generator_lora_path,
            current_discriminator_lora_path=current_discriminator_lora_path,
            replay_buffer=replay_buffer,
            round_idx=round_idx,
            external_val_records=external_val_records,
        )
        print(f"[Round] discriminator updated adapter saved to: {current_discriminator_lora_path}")
        save_replay_buffer_state(args, replay_buffer)

    # for round_idx in range(1, args.num_outer_rounds + 1):
    #     print(f"\n========== Outer Round {round_idx} / {args.num_outer_rounds} ==========")
    #
    #     print("[Round] incremental training discriminator ...")
    #     current_discriminator_lora_path = run_one_discriminator_incremental_round(
    #         real_records=real_records,
    #         args=args,
    #         current_generator_lora_path=current_generator_lora_path,
    #         current_discriminator_lora_path=current_discriminator_lora_path,
    #         replay_buffer=replay_buffer,
    #         round_idx=round_idx,
    #     )
    #     print(f"[Round] discriminator updated adapter saved to: {current_discriminator_lora_path}")
    #
    #     print("[Round] GRPO training generator ...")
    #     current_generator_lora_path = run_one_generator_grpo_round(
    #         real_records=real_records,
    #         args=args,
    #         current_generator_lora_path=current_generator_lora_path,
    #         current_discriminator_lora_path=current_discriminator_lora_path,
    #         round_idx=round_idx,
    #     )
    #     print(f"[Round] generator updated adapter saved to: {current_generator_lora_path}")

    summary = {
        "final_generator_lora_path": current_generator_lora_path,
        "final_discriminator_lora_path": current_discriminator_lora_path,
    }
    with open(os.path.join(args.output_dir, "final_paths.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] Training loop finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
