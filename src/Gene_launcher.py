import os
import sys
import json
import argparse
import datetime
import subprocess
import time
from typing import List


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_nodes", type=int, default=5)
    parser.add_argument("--min_edges", type=int, default=2)

    # Basic settings
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="abstrct")
    parser.add_argument("--few_shot", type=int, default=100)
    parser.add_argument("--train_script", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gene_SFT.py"))

    # Data paths
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--synthetic_train_file", type=str, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--synthetic_valid_file", type=str, default=None)

    # Pairing mode
    parser.add_argument("--pair_mode", type=str, default="index", choices=["index", "id"])

    # Model and output
    # Set the local model path here, or supply it on the command line.
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="./log_qwen_synth_paired")

    # Output format settings
    parser.add_argument("--output_language_field", type=str2bool, default=False)
    parser.add_argument("--keep_extra_top_fields", type=str2bool, default=False)

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])

    # Learning rate and epochs
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)

    # ========= LoRA / QLoRA =========
    parser.add_argument("--load_in_4bit", type=str2bool, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Checkpoint resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def get_dataset_lr():
    return {"aaec": 1e-4, "aaec_para": 1e-4, "aaec_essay": 1e-4,
            "abstrct": 5e-5, "cdcp": 1e-5}


def get_dataset_epochs_by_few_shot(few_shot: int):
    epochs = {5: 4, 100: 1}.get(few_shot)
    if epochs is None:
        raise ValueError("No automatic epoch setting for this few_shot; specify epochs explicitly.")
    return {name: epochs for name in ("aaec", "aaec_para", "aaec_essay", "abstrct", "cdcp")}


def run_command(cmd: List[str], env=None):
    print("Running command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    args = parse_args()

    dataset_name = args.dataset
    few_shot = args.few_shot
    gpu_id = args.gpu_id

    # Resolve bundled data relative to this script, independently of the working directory.
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    default_train_file = os.path.join(
        data_root, "data_mrp", f"{dataset_name}_{few_shot}percent", f"{dataset_name}_train.jsonl",
    )
    # Only 5% targets are currently bundled; other percentages require adding the matching file.
    default_synth_train_file = os.path.join(
        data_root, "Data_other", dataset_name, f"{dataset_name}_{few_shot}_gene.jsonl",
    )

    # Paired validation is optional; provide both source and synthetic validation files.
    default_valid_file = ""
    default_synth_valid_file = ""

    train_file = args.train_file if args.train_file is not None else default_train_file
    synthetic_train_file = (
        args.synthetic_train_file if args.synthetic_train_file is not None else default_synth_train_file
    )
    valid_file = args.valid_file if args.valid_file is not None else default_valid_file
    synthetic_valid_file = (
        args.synthetic_valid_file if args.synthetic_valid_file is not None else default_synth_valid_file
    )

    # Automatic learning rate and epochs
    dataset_lr = get_dataset_lr()
    dataset_epoch = get_dataset_epochs_by_few_shot(few_shot) if args.num_train_epochs is None else {}

    learning_rate = args.learning_rate if args.learning_rate is not None else dataset_lr.get(dataset_name, 1e-4)
    num_train_epochs = (
        args.num_train_epochs if args.num_train_epochs is not None else dataset_epoch.get(dataset_name, 3)
    )

    # Output directory
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_base = os.path.join(
        args.output_root,
        dataset_name,
        "train",
        f"{dataset_name}_{few_shot}percent",
        now_time,
    )
    os.makedirs(output_dir_base, exist_ok=True)

    # Save the resolved launcher configuration
    launcher_config = {
        "gpu_id": gpu_id,
        "dataset": dataset_name,
        "few_shot": few_shot,
        "train_script": args.train_script,
        "train_file": train_file,
        "synthetic_train_file": synthetic_train_file,
        "valid_file": valid_file,
        "synthetic_valid_file": synthetic_valid_file,
        "pair_mode": args.pair_mode,
        "model_path": args.model_path,
        "output_root": args.output_root,
        "output_language_field": args.output_language_field,
        "keep_extra_top_fields": args.keep_extra_top_fields,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seeds": args.seeds,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "timestamp": now_time,
        "min_nodes": args.min_nodes,
        "min_edges": args.min_edges,
    }

    with open(os.path.join(output_dir_base, "launcher_config.json"), "w", encoding="utf-8") as f:
        json.dump(launcher_config, f, ensure_ascii=False, indent=2)

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "disabled"

    # Launch one run per seed
    for seed in args.seeds:
        run_output_dir = os.path.join(output_dir_base, f"seed_{seed}_lr_{learning_rate}")
        os.makedirs(run_output_dir, exist_ok=True)

        run_config = {
            **launcher_config,
            "seed": seed,
            "seed_output_dir": run_output_dir,
        }
        with open(os.path.join(run_output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        cmd = [
            sys.executable, args.train_script,
            "--train_file", train_file,
            "--synthetic_train_file", synthetic_train_file,
            "--pair_mode", args.pair_mode,
            "--model_path", args.model_path,
            "--output_dir", run_output_dir,
            "--num_train_epochs", str(num_train_epochs),
            "--learning_rate", str(learning_rate),
            "--per_device_train_batch_size", str(args.per_device_train_batch_size),
            "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--max_length", str(args.max_length),
            "--warmup_ratio", str(args.warmup_ratio),
            "--weight_decay", str(args.weight_decay),
            "--logging_steps", str(args.logging_steps),
            "--save_steps", str(args.save_steps),
            "--save_total_limit", str(args.save_total_limit),
            "--lr_scheduler_type", str(args.lr_scheduler_type),
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--target_modules", args.target_modules,
            "--seed", str(seed),
            "--min_nodes", str(args.min_nodes),
            "--min_edges", str(args.min_edges),
        ]

        if args.output_language_field:
            cmd.append("--output_language_field")

        if args.keep_extra_top_fields:
            cmd.append("--keep_extra_top_fields")

        if args.load_in_4bit:
            cmd.append("--load_in_4bit")

        if args.resume_from_checkpoint is not None:
            cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

        # Forward validation paths when both source and target paths are supplied.
        if valid_file is not None and synthetic_valid_file is not None:
            cmd.extend(["--valid_file", valid_file])
            cmd.extend(["--synthetic_valid_file", synthetic_valid_file])

        print(f"=== Starting SFT training for seed {seed} ===")
        ret = run_command(cmd, env=env)

        if ret != 0:
            print(f"Training failed for seed={seed}.")
            continue

    print("All training runs have finished.")


if __name__ == "__main__":
    main()
