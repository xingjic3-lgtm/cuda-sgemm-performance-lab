import csv
import os
import re
import subprocess
from pathlib import Path


SIZES = [128, 256, 512, 1024, 2048, 4096]
RESULT_PATTERN = re.compile(
    r"Average elapsed time: \(\s*([0-9.]+)\) s, "
    r"performance: \(\s*([0-9.]+)\) GFLOPS"
)
FIELDS = [
    "record_type",
    "config_index",
    "num_threads",
    "bm",
    "bn",
    "bk",
    "wm",
    "wn",
    "wniter",
    "tm",
    "tn",
    "shape_type",
    "size_m",
    "size_n",
    "size_k",
    "kernel_time_s",
    "kernel_gflops",
    "cublas_time_s",
    "cublas_gflops",
    "relative_to_cublas",
    "status",
]


def run_sgemm(executable, kernel_id, size, environment):
    result = subprocess.run(
        [str(executable), str(kernel_id), str(size), str(size), str(size)],
        cwd=executable.parent,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"process_exit_{result.returncode}")

    match = RESULT_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError("timing_parse_failed")
    return float(match.group(1)), float(match.group(2))


def save_results(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"
    config_path = scripts_dir / "k10_sampled_configs.csv"
    result_path = project_root.parent / "phase3_tuning_results.csv"
    build_script = scripts_dir / "build_k10_config.bat"
    executable = project_root / "build-k10-autotune" / "sgemm.exe"

    with config_path.open(encoding="utf-8") as file:
        configs = list(csv.DictReader(file))

    with result_path.open(encoding="utf-8") as file:
        existing_rows = list(csv.DictReader(file))
    rows = [row for row in existing_rows if row["record_type"] == "phase1_baseline"]

    environment = os.environ.copy()
    cuda_bin = r"D:\anaconda3\envs\cuda-lab\Library\bin"
    environment["PATH"] = cuda_bin + os.pathsep + environment["PATH"]

    # Build the first configuration so cuBLAS is measured with the updated
    # warm-up code. These six cuBLAS measurements are reused for all configs.
    subprocess.run(
        ["cmd.exe", "/d", "/c", str(build_script), "1"],
        cwd=project_root,
        check=True,
    )
    cublas_results = {}
    for size in SIZES:
        print(f"\n[cuBLAS] size={size}")
        cublas_results[size] = run_sgemm(
            executable, 0, size, environment
        )

    for config_index, config in enumerate(configs, start=1):
        print(f"\n========== Config {config_index}/{len(configs)} ==========")
        if config_index != 1:
            try:
                subprocess.run(
                    [
                        "cmd.exe",
                        "/d",
                        "/c",
                        str(build_script),
                        str(config_index),
                    ],
                    cwd=project_root,
                    check=True,
                )
            except subprocess.CalledProcessError:
                for size in SIZES:
                    cublas_time, cublas_gflops = cublas_results[size]
                    rows.append(
                        {
                            "record_type": "phase3_tuning",
                            "config_index": config_index,
                            **config,
                            "shape_type": "square",
                            "size_m": size,
                            "size_n": size,
                            "size_k": size,
                            "kernel_time_s": "",
                            "kernel_gflops": "",
                            "cublas_time_s": cublas_time,
                            "cublas_gflops": cublas_gflops,
                            "relative_to_cublas": "",
                            "status": "compile_failed",
                        }
                    )
                save_results(result_path, rows)
                continue

        for size in SIZES:
            print(f"\n[Kernel 10] config={config_index}, size={size}")
            cublas_time, cublas_gflops = cublas_results[size]
            kernel_time = ""
            kernel_gflops = ""
            relative = ""
            status = "ok"
            try:
                kernel_time, kernel_gflops = run_sgemm(
                    executable, 10, size, environment
                )
                relative = kernel_gflops / cublas_gflops
            except RuntimeError as error:
                status = str(error)

            rows.append(
                {
                    "record_type": "phase3_tuning",
                    "config_index": config_index,
                    **config,
                    "shape_type": "square",
                    "size_m": size,
                    "size_n": size,
                    "size_k": size,
                    "kernel_time_s": kernel_time,
                    "kernel_gflops": kernel_gflops,
                    "cublas_time_s": cublas_time,
                    "cublas_gflops": cublas_gflops,
                    "relative_to_cublas": relative,
                    "status": status,
                }
            )

        # Save after every configuration so an interrupted run keeps progress.
        save_results(result_path, rows)

    print(f"\nCompleted {len(configs)} configurations.")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
