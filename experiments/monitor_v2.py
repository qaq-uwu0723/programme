"""Training monitor — refreshes every 3 minutes. Ctrl+C to stop."""
import time, os, subprocess, re
from datetime import datetime

LOG = "D:/programme/experiments/training_earlystop_v3.log"
STAGE1_SEC_PER_EP = 8.5   # ~28min / 200 epochs
STAGE2_SEC_PER_EP = 23.0  # ~116min / 300 epochs for V2.6; batch=32 similar

def gpu():
    try:
        return subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader",
            shell=True, text=True
        ).strip()
    except:
        return "N/A"

def parse_time(ts_str):
    """Parse '[HH:MM:SS]' to seconds since midnight."""
    m = re.search(r"(\d+):(\d+):(\d+)", ts_str)
    if m:
        return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))
    return None

def estimate_epoch(lines, now_sec, stage_keyword, target_epochs, sec_per_ep):
    """Estimate current epoch from elapsed time since stage start."""
    for l in lines:
        if stage_keyword in l:
            t = parse_time(l)
            if t:
                elapsed = now_sec - t
                ep = min(int(elapsed / sec_per_ep) + 1, target_epochs)
                return ep, elapsed
    return None, 0

while True:
    os.system("cls" if os.name == "nt" else "clear")
    now = datetime.now()
    now_sec = now.hour*3600 + now.minute*60 + now.second
    print("=" * 55)
    print(f"  Training Monitor  |  {now.strftime('%H:%M:%S')}")
    print("  batch=32, val/3ep, d=128, K=600, 500K FARAONIC")
    print("=" * 55)
    print(f"\n  GPU: {gpu()}")

    stage = "Unknown"
    ep_info = ""
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            lines = f.readlines()

        # Detect stage
        stage1_done = any("Done" in l and "Loss:" in l and any(c.isdigit() for c in l.split("Loss:")[1]) for l in lines)
        stage2_started = any("STAGE 2" in l for l in lines)
        final_done = any("FINAL" in l for l in lines)

        if final_done:
            stage = "COMPLETED"
        elif stage2_started:
            stage = "Stage 2: Diffusion"
            ep, elapsed = estimate_epoch(lines, now_sec, "STAGE 2", 300, STAGE2_SEC_PER_EP)
            if ep:
                remaining = (300 - ep) * STAGE2_SEC_PER_EP
                ep_info = f"Epoch ~{ep}/300  ({elapsed//60}min elapsed, ~{remaining//60}min remaining)"
        elif stage1_done:
            stage = "Stage 2: Diffusion (starting...)"
        else:
            stage = "Stage 1: Trend"
            ep, elapsed = estimate_epoch(lines, now_sec, "STAGE 1", 200, STAGE1_SEC_PER_EP)
            if ep:
                remaining = (200 - ep) * STAGE1_SEC_PER_EP
                ep_info = f"Epoch ~{ep}/200  ({elapsed//60}min elapsed, ~{remaining//60}min remaining)"

        print(f"\n  Stage: {stage}")
        if ep_info:
            bar_len = 20
            ep_num = int(ep_info.split("~")[1].split("/")[0]) if "~" in ep_info else 0
            total = 200 if "Trend" in stage else 300
            filled = int(bar_len * ep_num / total)
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            print(f"  {ep_info}")
            print(f"  {bar}")

        # Show key log lines
        for l in lines:
            l = l.strip()
            if "STAGE" in l or "Done" in l or "FINAL" in l or "Early stopping" in l:
                print(f"  {l}")
        for l in lines:
            l = l.strip()
            if "val:" in l:
                print(f"  {l}")

        print(f"\n  Last log: {lines[-1].strip() if lines else '...'}")
    else:
        print("\n  Waiting for log...")

    print(f"\n  [Next refresh: 3 min | Ctrl+C to exit]")
    time.sleep(180)
