"""Training monitor — auto-detects latest training log. Ctrl+C to stop.

Usage:
    python experiments/monitor_v2.py                      # auto-find latest
    python experiments/monitor_v2.py path/to/training.log # specific log
"""
import time, os, sys, subprocess, re, glob
from datetime import datetime


# --- Find log: CLI arg > latest file > fallback ---
def find_log():
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Search relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(script_dir)
    candidates = (
        glob.glob(os.path.join(project, "checkpoints", "exp*", "training.log")) +
        glob.glob(os.path.join(project, "experiments", "training*.log"))
    )
    if not candidates:
        # Fallback: search from cwd
        candidates = (
            glob.glob("checkpoints/exp*/training.log") +
            glob.glob("experiments/training*.log")
        )
    return max(candidates, key=os.path.getmtime) if candidates else None


LOG = find_log()
if not LOG or not os.path.exists(LOG):
    print("No training log found. Usage: python monitor_v2.py <log_path>")
    sys.exit(1)


def gpu():
    try:
        return subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw "
            "--format=csv,noheader",
            shell=True, text=True).strip()
    except Exception:
        return "N/A"


def parse_ts(line):
    """Extract seconds-from-midnight from [HH:MM:SS] timestamp."""
    m = re.search(r"\[(\d+):(\d+):(\d+)\]", line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def parse_epoch(line):
    """Extract (current_epoch, max_epoch) from progress lines."""
    m = re.search(r"(?:Trend|Diff) epoch\s+(\d+)/(\d+)", line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_losses(line):
    """Extract loss values from the Stage summary line."""
    # Matches: "120s | loss 0.0423->0.0012"
    m = re.search(r"loss\s+([\d.]+)->([\d.]+)", line)
    return (float(m.group(1)), float(m.group(2))) if m else None


last_size = 0
done = False
prev_stage = None

while True:
    os.system("cls" if os.name == "nt" else "clear")
    now = datetime.now()
    now_sec = now.hour * 3600 + now.minute * 60 + now.second

    # Detect file growth
    try:
        current_size = os.path.getsize(LOG)
    except OSError:
        current_size = 0
    growing = current_size > last_size
    last_size = current_size

    print("=" * 55)
    print(f"  TADS-ICS Monitor  |  {now.strftime('%H:%M:%S')}")
    print(f"  Log: {os.path.basename(LOG)}  |  {'[Growing]' if growing else '[Stale?]'}")
    print(f"  Path: {LOG}")
    print("=" * 55)
    print(f"\n  GPU: {gpu()}")

    if os.path.exists(LOG) and current_size > 0:
        with open(LOG, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]

        # --- Stage detection ---
        done = any("DONE" in l or "COMPLETE" in l for l in lines)
        stage2_started = any("STAGE 2: Diffusion" in l for l in lines)
        stage1_summary = any(("loss" in l and "->" in l and "Trend" not in l and
                              "STAGE 1" not in l and "STAGE 2" not in l and
                              "cont:" in l) for l in lines)
        stage1_started = any("STAGE 1: Trend" in l for l in lines)

        if done:
            stage = "COMPLETED"
        elif stage2_started:
            stage = "Stage 2: Diffusion"
        elif stage1_summary:
            stage = "Stage 2: Diffusion (starting...)"
        elif stage1_started:
            stage = "Stage 1: Trend"
        else:
            stage = "Initializing..."

        # --- Parse latest epoch progress ---
        # Find the newest progress line for current stage
        trend_ep = None
        diff_ep = None
        for l in lines:
            ep = parse_epoch(l)
            if ep is None:
                continue
            if "Trend" in l:
                trend_ep = ep
            elif "Diff" in l:
                diff_ep = ep

        ep_info = ""
        bar = "[" + "-" * 20 + "]"

        if "Trend" in stage or stage == "Initializing...":
            if trend_ep:
                cur, max_ep = trend_ep
                pct = cur / max_ep
                filled = int(20 * pct)
                bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
                ep_info = f"Epoch {cur}/{max_ep}"
        elif "Diffusion" in stage or done:
            if diff_ep:
                cur, max_ep = diff_ep
                pct = cur / max_ep
                filled = int(20 * pct)
                bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
                ep_info = f"Epoch {cur}/{max_ep}"
            elif trend_ep:
                # Stage 2 just started, no Diff progress yet
                ep_info = "Epoch 0/300"
            if done:
                ep_info = "Epoch max/max  (complete)"
                bar = "[" + "#" * 20 + "]"

        # --- Loss values from summary lines ---
        trend_loss_str = ""
        for l in lines:
            if "cont:" in l and "disc:" in l:
                # Stage 2 summary: "cont: 0.1234->0.0567  disc: 0.8901->0.3456"
                m = re.findall(r"([\d.]+)", l)
                if len(m) >= 4:
                    trend_loss_str = f"  cont: {m[0]}->{m[1]}  disc: {m[2]}->{m[3]}"
            elif "loss" in l and "->" in l and "Trend" not in l and "STAGE" not in l:
                if "cont" not in l and "disc" not in l and "s" in l:
                    # Stage 1 summary: "  120s | loss 0.0423->0.0012"
                    losses = parse_losses(l)
                    if losses:
                        trend_loss_str = f"  loss: {losses[0]:.4f}->{losses[1]:.4f}"

        # --- Display ---
        print(f"\n  Stage: {stage}")
        if ep_info:
            print(f"  {ep_info}")
            print(f"  {bar}")
        if trend_loss_str:
            print(f"  {trend_loss_str}")

        # Key events
        print(f"\n  --- Key Events ---")
        shown = 0
        for l in lines:
            s = l.strip()
            if any(kw in s for kw in [
                "STAGE", "Trend epoch", "Diff epoch",
                "Early stopping", "Restored best model",
                "Model saved", "KS  Train", "Max KS", "JSD",
                "DONE", "Done", "loss"  # catch summary lines
            ]):
                # Filter out duplicate/verbose lines
                if s in ("--- STAGE 1: Trend ---", "--- STAGE 2: Diffusion ---",
                         "--- Evaluation ---", "-- Model saved --", "=== DONE ==="):
                    print(f"  {s}")
                elif any(kw in s for kw in ["Trend epoch", "Diff epoch",
                                             "Early stopping", "Restored",
                                             "KS  Train", "Max KS", "JSD",
                                             "cont:", "disc:"]):
                    print(f"  {s}")
                shown += 1
        if shown == 0:
            print("  (no key events yet)")

        print(f"\n  Last: {lines[-1].strip() if lines else '...'}")
    else:
        print("\n  Waiting for log file...")

    if done:
        print(f"\n  {'=' * 45}")
        print(f"  TRAINING COMPLETE — exiting monitor.")
        print(f"  {'=' * 45}")
        break

    print(f"\n  [Next: 1 min | Ctrl+C to exit]")
    time.sleep(60)
