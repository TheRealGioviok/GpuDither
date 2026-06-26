#!/usr/bin/env python3

import os
import re
import csv
import math
import glob
import subprocess
from statistics import mean, stdev
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-images",
    action="store_true",
    help="Save generated output images.",
    default=False
)

args = parser.parse_args()

SAVE_OUTPUT_IMAGES = SAVE_OUTPUT_IMAGES = args.save_images

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow:")
    print("    pip install pillow")
    raise

# ============================================================
# Configuration
# ============================================================

EXECUTABLE = "./dither_gpu"

INPUT_DIR = "images"

OUTPUT_DIR = "benchmark_outputs"

MODES = [
    "threshold",
    "random",
    "bayer2",
    "bayer4",
    "bayer8",
    "floyd_steinberg_safe",
    "floyd_steinberg_vec",
    "floyd_steinberg_vec2",
]

BITS = 1

WARMUP_RUNS = 2
MEASURED_RUNS = 10

SUMMARY_CSV = "benchmark_summary.csv"
RAW_CSV = "benchmark_raw_runs.csv"
MODE_SUMMARY_CSV = "benchmark_mode_summary.csv"

TIME_RE = re.compile(r"kernel time\s+([0-9.]+)\s+ms")


def size_bucket(width, height):
    pixels = width * height

    if pixels < 512 * 512:
        return "small"

    if pixels <= 2048 * 2048:
        return "medium"

    return "large"


def confidence_interval(samples):
    n = len(samples)

    if n < 2:
        return 0.0

    s = stdev(samples)

    tcrit = t.ppf(0.975, n - 1)
    return tcrit * s / math.sqrt(n)


NULL_OUTPUT = os.devnull

def run_once(image_path, output_path, mode, bits):

    proc = subprocess.run(
        [
            EXECUTABLE,
            image_path,
            output_path,
            mode,
            str(bits)
        ],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"\nCommand failed:\n"
            f"{proc.stderr}\n"
        )

    match = TIME_RE.search(proc.stdout)

    if not match:
        raise RuntimeError(
            f"Could not parse kernel timing.\n"
            f"Output was:\n{proc.stdout}"
        )

    return float(match.group(1))


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    if not images:
        raise RuntimeError(
            f"No PNG images found in '{INPUT_DIR}'"
        )

    raw_rows = []
    summary_rows = []

    mode_aggregate = {}

    total_tests = len(images) * len(MODES)

    test_counter = 0

    for image_path in images:

        image_name = os.path.basename(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        pixels = width * height
        bucket = size_bucket(width, height)

        print()
        print("=" * 70)
        print(f"Image: {image_name}")
        print(f"Resolution: {width}x{height}")
        print(f"Pixels: {pixels:,}")
        print(f"Bucket: {bucket}")
        print("=" * 70)

        for mode in MODES:

            test_counter += 1

            print(
                f"[{test_counter}/{total_tests}] "
                f"{mode}"
            )

            base_name = os.path.splitext(image_name)[0]

            # Warmup
            for warmup in range(WARMUP_RUNS):

                warmup_out = (
                    os.path.join(
                        OUTPUT_DIR,
                        f"{base_name}_{mode}_warmup.png"
                    )
                    if SAVE_OUTPUT_IMAGES
                    else NULL_OUTPUT
                )

                run_once(
                    image_path,
                    warmup_out,
                    mode,
                    BITS
                )

            # Actual measure
            timings = []

            for run_idx in range(MEASURED_RUNS):

                out_file = (
                    os.path.join(
                        OUTPUT_DIR,
                        f"{base_name}_{mode}_run{run_idx}.png"
                    )
                    if SAVE_OUTPUT_IMAGES
                    else NULL_OUTPUT
                )

                runtime_ms = run_once(
                    image_path,
                    out_file,
                    mode,
                    BITS
                )

                timings.append(runtime_ms)

                raw_rows.append({
                    "image": image_name,
                    "width": width,
                    "height": height,
                    "pixels": pixels,
                    "bucket": bucket,
                    "mode": mode,
                    "run": run_idx,
                    "kernel_time_ms": runtime_ms,
                })

            avg = mean(timings)
            sd = stdev(timings) if len(timings) > 1 else 0.0
            ci95 = confidence_interval(timings)

            summary_rows.append({
                "image": image_name,
                "width": width,
                "height": height,
                "pixels": pixels,
                "bucket": bucket,
                "mode": mode,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "mean_ms": avg,
                "stddev_ms": sd,
                "ci95_ms": ci95,
                "min_ms": min(timings),
                "max_ms": max(timings),
            })

            mode_aggregate.setdefault(mode, []).extend(timings)

            print(
                f"    mean={avg:.4f} ms "
                f"± {ci95:.4f} ms (95% CI)"
            )

    # Write the csv files

    # all run data
    with open(RAW_CSV, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "width",
                "height",
                "pixels",
                "bucket",
                "mode",
                "run",
                "kernel_time_ms",
            ]
        )

        writer.writeheader()
        writer.writerows(raw_rows)

    # summary
    with open(SUMMARY_CSV, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "width",
                "height",
                "pixels",
                "bucket",
                "mode",
                "warmup_runs",
                "measured_runs",
                "mean_ms",
                "stddev_ms",
                "ci95_ms",
                "min_ms",
                "max_ms",
            ]
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # mode aggregate results
    mode_rows = []

    for mode, values in mode_aggregate.items():

        avg = mean(values)
        sd = stdev(values) if len(values) > 1 else 0.0
        ci95 = confidence_interval(values)

        mode_rows.append({
            "mode": mode,
            "samples": len(values),
            "mean_ms": avg,
            "stddev_ms": sd,
            "ci95_ms": ci95,
            "min_ms": min(values),
            "max_ms": max(values),
        })

    with open(MODE_SUMMARY_CSV, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "samples",
                "mean_ms",
                "stddev_ms",
                "ci95_ms",
                "min_ms",
                "max_ms",
            ]
        )

        writer.writeheader()
        writer.writerows(mode_rows)

    print()
    print("Benchmark complete.")
    print(f"Raw runs:      {RAW_CSV}")
    print(f"Per-image:     {SUMMARY_CSV}")
    print(f"Per-mode:      {MODE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()