import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TIMING_RE = re.compile(r"kernel time\s+([\d.]+)\s*ms", re.IGNORECASE)

CPU_MODE = "floyd_steinberg"
GPU_MODE = "floyd_steinberg_vec3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu",  default="./dither_cpu",          metavar="PATH",
                    help="path to dither_cpu binary (default: ./dither_cpu)")
    p.add_argument("--gpu",  default="./dither_gpu",          metavar="PATH",
                    help="path to dither_gpu binary (default: ./dither_gpu)")
    p.add_argument("--bits", default=2, type=int,             metavar="N",
                    help="quantisation bits 1-8 (default: 4)")
    p.add_argument("--runs", default=5, type=int,             metavar="N",
                    help="timed runs per size, excluding warm-up (default: 3)")
    p.add_argument("--out",  default="benchmark_results.csv", metavar="CSV",
                    help="output CSV path (default: benchmark_results.csv)")
    p.add_argument("--plot", action="store_true",
                    help="generate PNG charts (candlestick-style range plot + speedup plot)")
    p.add_argument("--plot-dir", default=".", metavar="DIR",
                    help="directory to save generated plots into (default: current dir)")
    return p.parse_args()


def check_binary(path: str, label: str) -> bool:
    if not Path(path).is_file():
        print(f"{label} binary not found: {path}")
        return False
    if not os.access(path, os.X_OK):
        print(f"{label} binary not executable: {path}")
        return False
    return True


def run_once(binary: str, mode: str, side: int, bits: int, tmp_out: str) -> float | None:
    cmd = [binary, "--gen", str(side), tmp_out, mode, str(bits)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"\ttimeout")
        return None
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        print(f"failed with code {result.returncode}): {result.stderr.strip()}")
        return None

    m = TIMING_RE.search(result.stdout)
    if not m:
        print(f"no time info parsed from: {result.stdout.strip()!r}")
        return None

    return float(m.group(1))


def benchmark(binary: str, mode: str, side: int, bits: int,
              runs: int, tmp_out: str) -> dict:
    # warmup
    run_once(binary, mode, side, bits, tmp_out)
    times = [t for _ in range(runs)
             if (t := run_once(binary, mode, side, bits, tmp_out)) is not None]
    if not times:
        return {"min": None, "max": None, "mean": None, "median": None,
                "stdev": None, "n": 0, "times": []}
    return {
        "min": min(times),
        "max": max(times),
        "mean": statistics.fmean(times),
        "median": statistics.median(times),
        "stdev": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "n": len(times),
        "times": times,
    }


def save_csv(path: str, rows: list[dict]):
    fields = ["side",
              "cpu_min_ms", "cpu_max_ms", "cpu_mean_ms", "cpu_median_ms", "cpu_stdev_ms", "cpu_n",
              "gpu_min_ms", "gpu_max_ms", "gpu_mean_ms", "gpu_median_ms", "gpu_stdev_ms", "gpu_n",
              "speedup"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved: {path}")


def make_plots(rows: list[dict], bits: int, plot_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    sides = [r["side"] for r in rows]
    x = list(range(len(sides)))

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.8,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    CPU_COLOR = "#2f6fed"   # blue
    GPU_COLOR = "#e0622f"   # orange
    WICK_COLOR = "#888888"

    def add_candles(ax, xpos, stats, color, offset, width=0.18):
        """Draw a candlestick: wick from min to max, body around mean +- stdev."""
        for xi, s, dx in zip(xpos, stats, [offset] * len(xpos)):
            if s["min"] is None:
                continue
            xc = xi + dx
            # wick: full min-max range
            ax.plot([xc, xc], [s["min"], s["max"]], color=WICK_COLOR,
                     linewidth=1.2, zorder=2, solid_capstyle="round")
            # body: mean +- stdev (clipped to min/max)
            lo = max(s["min"], s["mean"] - s["stdev"])
            hi = min(s["max"], s["mean"] + s["stdev"])
            if hi - lo < 1e-9:
                hi = lo + 1e-9
            ax.add_patch(plt.Rectangle((xc - width / 2, lo), width, hi - lo,
                                        facecolor=color, edgecolor=color,
                                        linewidth=0.8, alpha=0.9, zorder=3))
            # mean marker
            ax.plot(xc, s["mean"], marker="o", markersize=3.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.3, zorder=4)

    n = len(sides)
    fig_w = max(9, n * 1.0)
    candle_width = max(0.08, min(0.18, 1.6 / n))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5), dpi=150)

    cpu_stats = [{"min": r["cpu_min_ms"], "max": r["cpu_max_ms"],
                  "mean": r["cpu_mean_ms"], "stdev": r["cpu_stdev_ms"]} for r in rows]
    gpu_stats = [{"min": r["gpu_min_ms"], "max": r["gpu_max_ms"],
                  "mean": r["gpu_mean_ms"], "stdev": r["gpu_stdev_ms"]} for r in rows]

    add_candles(ax, x, cpu_stats, CPU_COLOR, offset=-candle_width * 0.7, width=candle_width)
    add_candles(ax, x, gpu_stats, GPU_COLOR, offset=candle_width * 0.7, width=candle_width)

    ax.set_yscale("log")
    ax.set_xticks(x)
    label_step = 1 if n <= 12 else (n + 11) // 12
    xticklabels = [f"{s}x{s}" if i % label_step == 0 else "" for i, s in enumerate(sides)]
    ax.set_xticklabels(xticklabels, rotation=45 if n > 7 else 0, ha="right" if n > 7 else "center")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xlabel("Image size")
    ax.set_ylabel("Kernel time (ms, log scale)")
    ax.set_title(f"Dithering kernel time — CPU vs GPU ({bits}-bit, mode: {CPU_MODE})",
                 fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CPU_COLOR, edgecolor=CPU_COLOR, alpha=0.9, label="CPU"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GPU_COLOR, edgecolor=GPU_COLOR, alpha=0.9, label="GPU"),
        plt.Line2D([0], [0], color=WICK_COLOR, linewidth=1.2, label="min–max range"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                    markeredgecolor="#444444", markersize=5, label="mean"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    fig.tight_layout()
    timing_path = os.path.join(plot_dir, "benchmark_timing.png")
    fig.savefig(timing_path)
    plt.close(fig)
    print(f"Plot saved: {timing_path}")

    fig2, ax2 = plt.subplots(figsize=(fig_w, 4.5), dpi=150)
    speedups = [r["speedup"] for r in rows]
    valid_x = [xi for xi, s in zip(x, speedups) if s is not None]
    valid_y = [s for s in speedups if s is not None]

    bar_width = max(0.3, min(0.5, 4.0 / n))
    bars = ax2.bar(valid_x, valid_y, width=bar_width, color=GPU_COLOR, alpha=0.9,
                    edgecolor=GPU_COLOR, zorder=3)
    ax2.axhline(1.0, color="#888888", linewidth=1, linestyle="--", zorder=2)

    label_fontsize = 9 if n <= 12 else max(6, 9 - (n - 12) // 4)
    for xi, yi in zip(valid_x, valid_y):
        ax2.text(xi, yi, f"{yi:.1f}x", ha="center", va="bottom",
                  fontsize=label_fontsize, color="#222222")

    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, rotation=45 if n > 7 else 0, ha="right" if n > 7 else "center")
    ax2.set_xlim(-0.7, n - 0.3)
    ax2.set_xlabel("Image size")
    ax2.set_ylabel("Speedup (CPU mean / GPU mean)")
    ax2.set_title("GPU speedup over CPU", fontsize=13, fontweight="bold", pad=14)

    fig2.tight_layout()
    speedup_path = os.path.join(plot_dir, "benchmark_speedup.png")
    fig2.savefig(speedup_path)
    plt.close(fig2)
    print(f"Plot saved: {speedup_path}")


def main():
    args = parse_args()
    cpu_ok = check_binary(args.cpu, "CPU")
    gpu_ok = check_binary(args.gpu, "GPU")

    if not cpu_ok or not gpu_ok:
        sys.exit("Wrong binaries")

    print(f"\nDithering benchmark")
    print(f"  cpu ({CPU_MODE}): {args.cpu}  ({'found' if cpu_ok else 'MISSING'})")
    print(f"  gpu ({GPU_MODE}): {args.gpu}  ({'found' if gpu_ok else 'MISSING'})")
    print(f"  bits: {args.bits}   runs per size: {args.runs} (+1 warm-up)\n")
    print(f"  {'size':^11}  {'cpu mean':>10}  {'cpu min/max':>16}  {'gpu mean':>10}  {'gpu min/max':>16}  {'speedup':>8}")
    print("  " + "-" * 90)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = os.path.join(tmpdir, "out.png")

        for side in SIZES:
            empty = {"min": None, "max": None, "mean": None, "median": None,
                      "stdev": None, "n": 0, "times": []}
            cpu = benchmark(args.cpu, CPU_MODE, side, args.bits, args.runs, tmp_out) \
                  if cpu_ok else empty
            gpu = benchmark(args.gpu, GPU_MODE, side, args.bits, args.runs, tmp_out) \
                  if gpu_ok else empty

            speedup = (cpu["mean"] / gpu["mean"]
                       if cpu["mean"] and gpu["mean"] else None)

            cpu_mean_str = f"{cpu['mean']:>9.2f} ms" if cpu["mean"] else "       N/A"
            gpu_mean_str = f"{gpu['mean']:>9.2f} ms" if gpu["mean"] else "       N/A"
            cpu_mm_str = f"{cpu['min']:.2f}/{cpu['max']:.2f}" if cpu["min"] is not None else "N/A"
            gpu_mm_str = f"{gpu['min']:.2f}/{gpu['max']:.2f}" if gpu["min"] is not None else "N/A"
            spd_str = f"{speedup:>7.2f}x" if speedup else "       N/A"
            print(f"  {side:>5}x{side:<5}  {cpu_mean_str}  {cpu_mm_str:>16}  {gpu_mean_str}  {gpu_mm_str:>16}  {spd_str}")

            rows.append({
                "side":          side,
                "cpu_min_ms":    cpu["min"],
                "cpu_max_ms":    cpu["max"],
                "cpu_mean_ms":   cpu["mean"],
                "cpu_median_ms": cpu["median"],
                "cpu_stdev_ms":  cpu["stdev"],
                "cpu_n":         cpu["n"],
                "gpu_min_ms":    gpu["min"],
                "gpu_max_ms":    gpu["max"],
                "gpu_mean_ms":   gpu["mean"],
                "gpu_median_ms": gpu["median"],
                "gpu_stdev_ms":  gpu["stdev"],
                "gpu_n":         gpu["n"],
                "speedup":       speedup,
            })

    save_csv(args.out, rows)

    if args.plot:
        make_plots(rows, args.bits, args.plot_dir)


if __name__ == "__main__":
    main()