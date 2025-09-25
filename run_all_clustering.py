import os
import subprocess

# Folder containing clustering scripts
folder = os.path.dirname(os.path.abspath(__file__))

# Output file to save results
output_file = os.path.join(folder, "clustering_results.txt")

# Find all clustering scripts (assuming they have 'cluster' in their name and are .py files)
scripts = [
    f for f in os.listdir(folder)
    if f.endswith('.py')
    and 'cluster' in f
    and 'run' in f
    and f != os.path.basename(__file__)
]

# Predefined argument mappings for specific clustering scripts
# If a script name (without .py) is present here, these arguments will be used.
ARG_MAP: dict[str, list[str]] = {
    # mushroom clustering
    "run_mushroom_clustering": [
        "--selector", "mrmr",
        "--mrmr_alpha", "0.5",
        "--keep_top", "6",
        "--method", "bmm",
        "--restarts", "100",
        "--daem",
        "--seed", "42",
    ],
    # bank clustering
    "run_bank_clustering": [
        "--variant", "full",
        "--drop_duration",
        "--selector", "mrmr",
        "--mrmr_alpha", "0.5",
        "--keep_top", "8",
        "--method", "auto",
        "--overspec", "6",
        "--restarts", "120",
        "--daem",
        "--min_pi", "0.10",
        "--weight_bits",
        "--trim_em",
        "--tau_hi", "0.92",
        "--seed", "7",
    ],
    # HAR clustering
    "run_har_clustering": [
        "--selector", "mi",
        "--keep_top", "128",
        "--pca", "256",
        "--method", "auto",
        "--overspec", "20",
        "--cov_type", "full",
        "--restarts", "25",
        "--seed", "7",
    ],
}

print(scripts)


with open(output_file, "w") as out_f:
    for script in sorted(scripts):

        dataset_name = script.replace('.py', '')
        if dataset_name == 'run_all_clustering':
            continue  # Skip this script itself
        if 'cifar' in dataset_name.lower():
            continue  # Skip CIFAR datasets

        # Build command
        cmd = ["python3", os.path.join(folder, script)]
        if dataset_name in ARG_MAP:
            cmd.extend(ARG_MAP[dataset_name])

        out_f.write(f"===== Results for {dataset_name} =====\n\n")
        out_f.write(f"Command: {' '.join(cmd)}\n\n")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, check=True
            )
            out_f.write(result.stdout)
        except subprocess.CalledProcessError as e:
            out_f.write(f"Error running {script}:\n{e.stderr}\n")
        out_f.write("\n\n\n\n\n")  # Large spaces after each result

print(f"All clustering results saved to {output_file}")