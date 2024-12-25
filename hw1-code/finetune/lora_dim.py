import subprocess

# Define LoRA ranks to iterate over
lora_ranks = [1, 2, 4, 8, 16, 32]

# Base command with placeholders for dynamic values
base_command = [
    "python", "train.py",
    "--model_name_or_path", "gpt2",
    "--max_length", "512",
    "--trust_remote_code", "True",
    "--use_lora", "True",
    "--lora_scaling", "32",  # Assuming scaling remains constant
    "--lora_module_name", "h.",
    "--data_path", "./data/alpaca_data.json",
    "--epochs", "4",
    "--train_batch_size", "8",
    "--gradient_accumulation_steps", "4",
    "--lr", "3e-4",
    "--lr_warmup_ratio", "0.03",
    "--weight_decay", "0.01",
    "--seed", "42",
    "--eval_batch_size", "16",
    "--eval_ratio", "0.01",
    "--eval_interval", "100"
]

# Iterate over each LoRA rank
for rank in lora_ranks:
    # Update the LoRA dimension in the command
    command = base_command + ["--lora_dim", str(rank)]
    
    # Update the output directory to reflect the current LoRA rank
    output_dir = f"gpt2-alpaca-lora-rank-{rank}"
    command += ["--output_dir_name", output_dir]
    
    # Print the command to be executed
    print("Running command:", " ".join(command))
    
    # Execute the command
    subprocess.run(command)
