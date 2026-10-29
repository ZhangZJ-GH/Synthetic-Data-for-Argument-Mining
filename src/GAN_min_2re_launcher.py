import os
import sys
import json
import argparse
import datetime
import subprocess
import time
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--min_nodes", type=int, default=5)
    parser.add_argument("--min_edges", type=int, default=2)

    # Dataset settings for paths and log directories
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="abstrct")
    parser.add_argument("--few_shot", type=int, default=100)
    parser.add_argument("--train_script", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAN_min_2re.py"))
    parser.add_argument("--output_root", type=str, default="./log_joint_grpo_loop")
    parser.add_argument("--resume_output_dir", type=str, default=None, help="Existing output_dir to continue from; if omitted, create a new timestamped output_dir.")

    # Data paths with optional overrides
    parser.add_argument("--real_file", type=str, default=None)
    parser.add_argument("--disc_external_val_file", type=str)
    parser.add_argument("--disc_external_val_fixed_candidates", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--text_key", type=str, default="input")

    # Model path options
    # Set local model and adapter paths here; CLI paths take precedence.
    parser.add_argument("--generator_base_model_path", type=str, default="")
    parser.add_argument("--generator_init_lora_path", type=str, default="")
    parser.add_argument("--discriminator_base_model_path", type=str, default="")
    parser.add_argument("--discriminator_init_lora_path", type=str, default="")

    # Devices and precision
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generator_load_in_4bit", action="store_true")
    parser.add_argument("--discriminator_load_in_4bit", action="store_true")
    parser.add_argument("--generator_device", type=str, default=None)
    parser.add_argument("--discriminator_device", type=str, default=None)

    # Outer training rounds
    parser.add_argument("--num_outer_rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # ===== GRPO =====
    parser.add_argument("--grpo_num_generations", type=int, default=3)
    parser.add_argument("--grpo_per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--grpo_learning_rate", type=float, default=1e-5)
    parser.add_argument("--grpo_num_train_epochs", type=float, default=2.0)
    parser.add_argument("--grpo_max_steps", type=int, default=-1)
    parser.add_argument("--grpo_logging_steps", type=int, default=10)
    parser.add_argument("--grpo_save_steps", type=int, default=200)
    parser.add_argument("--grpo_save_total_limit", type=int, default=3)
    parser.add_argument("--grpo_resume_from_checkpoint", type=str, default="auto", help="auto / none / explicit checkpoint path")
    parser.add_argument("--grpo_save_only_model", action="store_true", help="Only save model weights; disables exact optimizer-state resume.")
    parser.add_argument("--grpo_max_completion_length", type=int, default=10000)
    parser.add_argument("--grpo_temperature", type=float, default=0.9)
    parser.add_argument("--grpo_top_p", type=float, default=0.9)
    parser.add_argument("--grpo_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--grpo_beta", type=float, default=0.0)
    parser.add_argument("--grpo_epsilon", type=float, default=0.2)
    parser.add_argument("--grpo_num_iterations", type=int, default=2)
    parser.add_argument("--grpo_scale_rewards", type=str, default="batch")
    parser.add_argument("--grpo_loss_type", type=str, default="dapo")

    # ===== reward =====
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="neg_log_p_real",
        choices=["neg_log_p_real", "zero_one"],
    )
    parser.add_argument("--grpo_discriminator_reward_weight", type=float, default=1.0)
    parser.add_argument("--grpo_exact_match_reward_weight", type=float, default=0.5)

    # Incremental discriminator training
    parser.add_argument("--discriminator_max_length", type=int, default=32768)
    parser.add_argument("--disc_num_permutations_per_group", type=int, default=2)
    parser.add_argument("--disc_replay_ratio", type=float, default=0.5)
    parser.add_argument("--disc_replay_buffer_max_size", type=int, default=50000)
    parser.add_argument("--disc_val_ratio", type=float, default=0.1,
                        help="Internal validation fraction; 0 selects the explicit external validation file.")
    parser.add_argument("--disc_per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--disc_per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--disc_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--disc_learning_rate", type=float, default=1e-5)
    parser.add_argument("--disc_weight_decay", type=float, default=0.01)
    parser.add_argument("--disc_num_train_epochs", type=float, default=8)
    parser.add_argument("--disc_max_steps", type=int, default=-1)
    parser.add_argument("--disc_logging_steps", type=int, default=10)
    parser.add_argument("--disc_eval_steps", type=int, default=0)
    parser.add_argument("--disc_save_steps", type=int, default=0)

    # Sample discriminator data with the updated generator
    parser.add_argument("--generator_prompt_max_length", type=int, default=16384)
    parser.add_argument("--generator_sample_max_new_tokens", type=int, default=16384)
    parser.add_argument("--generator_sample_temperature", type=float, default=0.9)
    parser.add_argument("--generator_sample_top_p", type=float, default=0.9)
    parser.add_argument("--generator_sample_repetition_penalty", type=float, default=1.05)
    parser.add_argument("--generator_sampling_batch_size", type=int, default=4)
    parser.add_argument("--generator_sampling_round_candidates", type=int, default=3)


    parser.add_argument("--no_skip_completed_rounds", dest="skip_completed_rounds", action="store_false")
    parser.set_defaults(skip_completed_rounds=True)

    return parser.parse_args()


def run_command(cmd: List[str], env=None, log_file: str = None):
    print("Running command:")
    print(" ".join(cmd))

    if log_file is None:
        result = subprocess.run(cmd, env=env)
        return result.returncode

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"[START] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 120 + "\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        ret = process.wait()

        f.write("\n" + "=" * 120 + "\n")
        f.write(f"[END] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | return_code={ret}\n")
        f.write("=" * 120 + "\n")
        f.flush()

    return ret


def main():
    # time.sleep(3600)
    args = parse_args()

    dataset_name = args.dataset
    few_shot = args.few_shot
    gpu_id = args.gpu_id

    # Resolve bundled data relative to this script, independently of the working directory.
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
    default_real_file = os.path.join(
        data_root, "data_mrp", f"{dataset_name}_{few_shot}percent", f"{dataset_name}_train.jsonl",
    )
    real_file = args.real_file if args.real_file is not None else default_real_file
    # Default: hold out 10% internally. To use an external set, pass BOTH
    # --disc_val_ratio 0 and --disc_external_val_file /path/to/validation.jsonl.
    disc_external_val_file = args.disc_external_val_file
    generator_base_model_path = args.generator_base_model_path
    generator_init_lora_path = args.generator_init_lora_path
    discriminator_base_model_path = args.discriminator_base_model_path
    discriminator_init_lora_path = args.discriminator_init_lora_path

    # Output directory
    if args.resume_output_dir is not None:
        output_dir = args.resume_output_dir
        now_time = os.path.basename(os.path.normpath(output_dir))
        print(f"[Resume] Reusing existing output_dir: {output_dir}")
    else:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(
            args.output_root,
            dataset_name,
            "train",
            f"{dataset_name}_{few_shot}percent",
            now_time,
        )
    os.makedirs(output_dir, exist_ok=True)
    terminal_log_file = os.path.join(output_dir, "train_terminal.log")

    # Save the resolved launcher configuration
    # "text_key": args.text_key,
    launcher_config = {
        "gpu_id": gpu_id,
        "dataset": dataset_name,
        "few_shot": few_shot,
        "train_script": args.train_script,
        "real_file": real_file,
        "disc_external_val_file": disc_external_val_file,
        "output_root": args.output_root,
        "resume_output_dir": args.resume_output_dir,
        "final_output_dir": output_dir,
        "generator_base_model_path": generator_base_model_path,
        "generator_init_lora_path": generator_init_lora_path,
        "discriminator_base_model_path": discriminator_base_model_path,
        "discriminator_init_lora_path": discriminator_init_lora_path,
        "bf16": args.bf16,
        "generator_load_in_4bit": args.generator_load_in_4bit,
        "discriminator_load_in_4bit": args.discriminator_load_in_4bit,
        "generator_device": args.generator_device,
        "discriminator_device": args.discriminator_device,
        "num_outer_rounds": args.num_outer_rounds,
        "seed": args.seed,
        "grpo_num_generations": args.grpo_num_generations,
        "grpo_per_device_train_batch_size": args.grpo_per_device_train_batch_size,
        "grpo_gradient_accumulation_steps": args.grpo_gradient_accumulation_steps,
        "grpo_learning_rate": args.grpo_learning_rate,
        "grpo_num_train_epochs": args.grpo_num_train_epochs,
        "grpo_max_steps": args.grpo_max_steps,
        "grpo_logging_steps": args.grpo_logging_steps,
        "grpo_save_steps": args.grpo_save_steps,
        "grpo_save_total_limit": args.grpo_save_total_limit,
        "grpo_resume_from_checkpoint": args.grpo_resume_from_checkpoint,
        "grpo_save_only_model": args.grpo_save_only_model,
        "grpo_max_completion_length": args.grpo_max_completion_length,
        "grpo_temperature": args.grpo_temperature,
        "grpo_top_p": args.grpo_top_p,
        "grpo_repetition_penalty": args.grpo_repetition_penalty,
        "grpo_beta": args.grpo_beta,
        "grpo_epsilon": args.grpo_epsilon,
        "grpo_num_iterations": args.grpo_num_iterations,
        "grpo_scale_rewards": args.grpo_scale_rewards,
        "grpo_loss_type": args.grpo_loss_type,
        "reward_mode": args.reward_mode,
        "grpo_discriminator_reward_weight": args.grpo_discriminator_reward_weight,
        "grpo_exact_match_reward_weight": args.grpo_exact_match_reward_weight,
        "discriminator_max_length": args.discriminator_max_length,
        "disc_num_permutations_per_group": args.disc_num_permutations_per_group,
        "disc_replay_ratio": args.disc_replay_ratio,
        "disc_replay_buffer_max_size": args.disc_replay_buffer_max_size,
        "disc_val_ratio": args.disc_val_ratio,
        "disc_external_val_fixed_candidates": args.disc_external_val_fixed_candidates,
        "disc_per_device_train_batch_size": args.disc_per_device_train_batch_size,
        "disc_per_device_eval_batch_size": args.disc_per_device_eval_batch_size,
        "disc_gradient_accumulation_steps": args.disc_gradient_accumulation_steps,
        "disc_learning_rate": args.disc_learning_rate,
        "disc_weight_decay": args.disc_weight_decay,
        "disc_num_train_epochs": args.disc_num_train_epochs,
        "disc_max_steps": args.disc_max_steps,
        "disc_logging_steps": args.disc_logging_steps,
        "disc_eval_steps": args.disc_eval_steps,
        "disc_save_steps": args.disc_save_steps,
        "generator_prompt_max_length": args.generator_prompt_max_length,
        "generator_sample_max_new_tokens": args.generator_sample_max_new_tokens,
        "generator_sample_temperature": args.generator_sample_temperature,
        "generator_sample_top_p": args.generator_sample_top_p,
        "generator_sample_repetition_penalty": args.generator_sample_repetition_penalty,
        "generator_sampling_batch_size": args.generator_sampling_batch_size,
        "generator_sampling_round_candidates": args.generator_sampling_round_candidates,
        "timestamp": now_time,
        "disable_thinking": args.disable_thinking,
        "min_nodes": args.min_nodes,
        "min_edges": args.min_edges,
        "terminal_log_file": terminal_log_file,
        "disable_internal_tee": True,
        "skip_completed_rounds": args.skip_completed_rounds,
    }

    with open(os.path.join(output_dir, "launcher_config.json"), "w", encoding="utf-8") as f:
        json.dump(launcher_config, f, ensure_ascii=False, indent=2)

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "disabled"
    env["DISABLE_INTERNAL_TEE"] = "1"

    # "--text_key", args.text_key,
    # Build the training command
    cmd = [
        sys.executable, args.train_script,
        "--real_file", real_file,
        "--output_dir", output_dir,
        "--generator_base_model_path", generator_base_model_path,
        "--generator_init_lora_path", generator_init_lora_path,
        "--discriminator_base_model_path", discriminator_base_model_path,
        "--discriminator_init_lora_path", discriminator_init_lora_path,
        "--num_outer_rounds", str(args.num_outer_rounds),
        "--seed", str(args.seed),
        "--grpo_num_generations", str(args.grpo_num_generations),
        "--grpo_per_device_train_batch_size", str(args.grpo_per_device_train_batch_size),
        "--grpo_gradient_accumulation_steps", str(args.grpo_gradient_accumulation_steps),
        "--grpo_learning_rate", str(args.grpo_learning_rate),
        "--grpo_num_train_epochs", str(args.grpo_num_train_epochs),
        "--grpo_max_steps", str(args.grpo_max_steps),
        "--grpo_logging_steps", str(args.grpo_logging_steps),
        "--grpo_save_steps", str(args.grpo_save_steps),
        "--grpo_save_total_limit", str(args.grpo_save_total_limit),
        "--grpo_resume_from_checkpoint", args.grpo_resume_from_checkpoint,
        "--grpo_max_completion_length", str(args.grpo_max_completion_length),
        "--grpo_temperature", str(args.grpo_temperature),
        "--grpo_top_p", str(args.grpo_top_p),
        "--grpo_repetition_penalty", str(args.grpo_repetition_penalty),
        "--grpo_beta", str(args.grpo_beta),
        "--grpo_epsilon", str(args.grpo_epsilon),
        "--grpo_num_iterations", str(args.grpo_num_iterations),
        "--grpo_scale_rewards", args.grpo_scale_rewards,
        "--grpo_loss_type", args.grpo_loss_type,
        "--reward_mode", args.reward_mode,
        "--grpo_discriminator_reward_weight", str(args.grpo_discriminator_reward_weight),
        "--grpo_exact_match_reward_weight", str(args.grpo_exact_match_reward_weight),
        "--discriminator_max_length", str(args.discriminator_max_length),
        "--disc_num_permutations_per_group", str(args.disc_num_permutations_per_group),
        "--disc_replay_ratio", str(args.disc_replay_ratio),
        "--disc_replay_buffer_max_size", str(args.disc_replay_buffer_max_size),
        "--disc_val_ratio", str(args.disc_val_ratio),
        "--disc_per_device_train_batch_size", str(args.disc_per_device_train_batch_size),
        "--disc_per_device_eval_batch_size", str(args.disc_per_device_eval_batch_size),
        "--disc_gradient_accumulation_steps", str(args.disc_gradient_accumulation_steps),
        "--disc_learning_rate", str(args.disc_learning_rate),
        "--disc_weight_decay", str(args.disc_weight_decay),
        "--disc_num_train_epochs", str(args.disc_num_train_epochs),
        "--disc_max_steps", str(args.disc_max_steps),
        "--disc_logging_steps", str(args.disc_logging_steps),
        "--disc_eval_steps", str(args.disc_eval_steps),
        "--disc_save_steps", str(args.disc_save_steps),
        "--generator_prompt_max_length", str(args.generator_prompt_max_length),
        "--generator_sample_max_new_tokens", str(args.generator_sample_max_new_tokens),
        "--generator_sample_temperature", str(args.generator_sample_temperature),
        "--generator_sample_top_p", str(args.generator_sample_top_p),
        "--generator_sample_repetition_penalty", str(args.generator_sample_repetition_penalty),
        "--generator_sampling_batch_size", str(args.generator_sampling_batch_size),
        "--generator_sampling_round_candidates", str(args.generator_sampling_round_candidates),
        "--min_nodes", str(args.min_nodes),
        "--min_edges", str(args.min_edges),
    ]


    if args.bf16:
        cmd.append("--bf16")
    if args.generator_load_in_4bit:
        cmd.append("--generator_load_in_4bit")
    if args.discriminator_load_in_4bit:
        cmd.append("--discriminator_load_in_4bit")
    if args.generator_device is not None:
        cmd.extend(["--generator_device", args.generator_device])
    if args.discriminator_device is not None:
        cmd.extend(["--discriminator_device", args.discriminator_device])
    if args.disc_external_val_fixed_candidates:
        cmd.append("--disc_external_val_fixed_candidates")
    if args.grpo_save_only_model:
        cmd.append("--grpo_save_only_model")
    if not args.skip_completed_rounds:
        cmd.append("--no_skip_completed_rounds")
    if args.disable_thinking:
        cmd.append("--disable_thinking")
    if disc_external_val_file is not None:
        cmd.extend(["--disc_external_val_file", disc_external_val_file])

    ret = run_command(cmd, env=env, log_file=terminal_log_file)

    if ret != 0:
        print("Training failed.")
        raise SystemExit(ret)

    print("Training completed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
