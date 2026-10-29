import os
import sys
import json
import argparse
import datetime
import subprocess
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

    # Dataset settings used for paths and log directories
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="abstrct")
    parser.add_argument("--few_shot", type=int, default=100)

    # Optional overrides for default data paths
    parser.add_argument("--real_file", type=str, default=None)
    parser.add_argument("--synthetic_file", type=str, default=None)

    # Training script and model
    parser.add_argument("--train_script", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Disc_SFT.py"))
    # Set the local model path here, or supply it on the command line.
    parser.add_argument("--model_name_or_path", type=str, default="")

    # Output root directory
    parser.add_argument("--output_root", type=str, default="./log_discriminator_new")

    # Training settings forwarded to the training script
    parser.add_argument("--text_key", type=str, default="input")
    parser.add_argument("--max_length", type=int, default=32768)

    parser.add_argument("--train_permutations_per_group", type=int, default=6)
    parser.add_argument("--eval_permutations_per_group", type=int, default=24)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
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


def run_command(cmd: List[str], env=None):
    print("Running command:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def get_dataset_epochs_by_few_shot(few_shot: int):
    epochs = {5: 8, 100: 2}.get(few_shot)
    if epochs is None:
        raise ValueError("No automatic epoch setting for this few_shot; specify epochs explicitly.")
    return {name: epochs for name in ("aaec", "aaec_para", "aaec_essay", "abstrct", "cdcp")}


def get_dataset_lr():
    return {"aaec": 1e-4, "aaec_para": 1e-4, "aaec_essay": 1e-4,
            "abstrct": 5e-5, "cdcp": 1e-5}


def main():
    args = parse_args()

    gpu_id = args.gpu_id
    dataset_name = args.dataset
    few_shot = args.few_shot

    # Resolve bundled data relative to this script, independently of the working directory.
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    default_real_file = os.path.join(
        data_root, "data_mrp", f"{dataset_name}_{few_shot}percent", f"{dataset_name}_train.jsonl",
    )
    # Discriminator synthetic groups are not bundled yet; supply --synthetic_file.
    default_synthetic_file = ""

    real_file = args.real_file if args.real_file is not None else default_real_file
    synthetic_file = args.synthetic_file if args.synthetic_file is not None else default_synthetic_file

    # Automatic epochs and learning rate with explicit overrides
    dataset_lr = get_dataset_lr()
    auto_epochs = (get_dataset_epochs_by_few_shot(few_shot).get(dataset_name, 2)
                   if args.epochs is None else args.epochs)
    final_epochs = args.epochs if args.epochs is not None else auto_epochs
    final_lr = (args.learning_rate if args.learning_rate is not None
                else dataset_lr.get(dataset_name, 1e-4))

    # Output directory grouped by dataset and data percentage
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        args.output_root,
        dataset_name,
        "train",
        f"{dataset_name}_{few_shot}percent",
        now_time,
    )
    os.makedirs(output_dir, exist_ok=True)

    launcher_config = {
        "gpu_id": gpu_id,
        "dataset": dataset_name,
        "few_shot": few_shot,
        "train_script": args.train_script,
        "real_file": real_file,
        "synthetic_file": synthetic_file,
        "model_name_or_path": args.model_name_or_path,
        "output_root": args.output_root,
        "final_output_dir": output_dir,
        "text_key": args.text_key,
        "max_length": args.max_length,
        "train_permutations_per_group": args.train_permutations_per_group,
        "eval_permutations_per_group": args.eval_permutations_per_group,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "epochs": final_epochs,
        "learning_rate": final_lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "use_lora": args.use_lora,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "timestamp": now_time,
    }

    with open(os.path.join(output_dir, "launcher_config.json"), "w", encoding="utf-8") as f:
        json.dump(launcher_config, f, ensure_ascii=False, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "disabled"

    cmd = [
        sys.executable, args.train_script,
        "--real_file", real_file,
        "--synthetic_file", synthetic_file,
        "--text_key", args.text_key,
        "--output_dir", output_dir,
        "--model_name_or_path", args.model_name_or_path,
        "--max_length", str(args.max_length),
        "--train_permutations_per_group", str(args.train_permutations_per_group),
        "--eval_permutations_per_group", str(args.eval_permutations_per_group),
        "--val_ratio", str(args.val_ratio),
        "--seed", str(args.seed),
        "--epochs", str(final_epochs),
        "--learning_rate", str(final_lr),
        "--weight_decay", str(args.weight_decay),
        "--train_batch_size", str(args.train_batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--logging_steps", str(args.logging_steps),
        "--save_total_limit", str(args.save_total_limit),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--target_modules", args.target_modules,
    ]

    if args.bf16:
        cmd.append("--bf16")
    if args.use_lora:
        cmd.append("--use_lora")
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")

    ret = run_command(cmd, env=env)

    if ret != 0:
        print("Training failed.")
        raise SystemExit(ret)

    print("Training completed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
