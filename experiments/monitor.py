"""Real-time training monitor — reads log file and GPU stats every few seconds."""
import time, os, subprocess, sys

LOG = "D:/programme/experiments/training_earlystop.log"

def gpu_stats():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader",
            shell=True, text=True
        ).strip()
        return out
    except:
        return "GPU: N/A"

def main():
    last_line = ""
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 55)
        print("  Mask-DDPM Training Monitor")
        print(f"  Log: {LOG}")
        print("=" * 55)

        # GPU
        print(f"\n  GPU: {gpu_stats()}")

        # Log content
        if os.path.exists(LOG):
            with open(LOG, encoding="utf-8") as f:
                lines = f.readlines()

            # Summary
            for line in lines:
                if "STAGE 1" in line:
                    print(f"\n  {line.strip()}")
                if "STAGE 2" in line:
                    print(f"\n  {line.strip()}")
                if "Done" in line and "Loss:" in line:
                    print(f"  {line.strip()}")
                if "Diff epoch" in line:
                    parts = line.strip().split()
                    ep = parts[2] if len(parts) > 2 else "?"
                    cont = parts[4] if len(parts) > 4 else "?"
                    disc = parts[6] if len(parts) > 6 else "?"
                    val = parts[8] if len(parts) > 8 else "?"
                    print(f"\n  Latest epoch: {ep}")
                    print(f"    cont={cont}  disc={disc}  val={val}")
                if "Early stopping" in line:
                    print(f"\n  *** {line.strip()} ***")
                if "RESULTS:" in line:
                    print(f"\n  {'='*45}")
                    print(f"  {line.strip()}")

            # Show last 3 lines
            print(f"\n  --- Last lines ---")
            for line in lines[-3:]:
                print(f"  {line.rstrip()}")
        else:
            print("\n  Log file not found — training may not have started yet")

        print(f"\n  [Refresh: {time.strftime('%H:%M:%S')} | Ctrl+C to exit]")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
