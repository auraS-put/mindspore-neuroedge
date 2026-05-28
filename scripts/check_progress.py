"""Poll OBS for benchmark progress and display live metrics.

Usage:
    python scripts/check_progress.py <session_id>
    python scripts/check_progress.py <session_id> --watch    # poll every 30s
    python scripts/check_progress.py --latest                # find most recent session
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(".env")


def get_obs_client():
    from obs import ObsClient
    return ObsClient(
        access_key_id=os.environ["HUAWEI_AK"],
        secret_access_key=os.environ["HUAWEI_SK"],
        server=f"https://obs.{os.environ['HUAWEI_REGION']}.myhuaweicloud.com",
    )


def find_latest_session(obs):
    """Find the most recent benchmark session by listing output/ prefixes."""
    resp = obs.listObjects("auras-experiments", prefix="output/benchmark_", delimiter="/")
    if resp.status != 200:
        print(f"Error listing sessions: {resp.status}")
        return None
    prefixes = [cp.prefix for cp in (resp.body.commonPrefixs or [])]
    if not prefixes:
        print("No benchmark sessions found.")
        return None
    # Sort by name (timestamp-based) and pick latest
    prefixes.sort()
    latest = prefixes[-1].rstrip("/").split("benchmark_")[1]
    return latest


def download_progress(obs, session_id):
    """Download progress.json from OBS."""
    key = f"output/benchmark_{session_id}/progress.json"
    resp = obs.getObject("auras-experiments", key, loadStreamInMemory=True)
    if resp.status != 200:
        return None
    return json.loads(resp.body.buffer.decode("utf-8"))


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def display_progress(progress):
    """Pretty-print benchmark progress."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK SESSION: {progress['session_id']}")
    print(f"  Status: {progress['status'].upper()}")
    print(f"  Mode: {progress['run_mode']}  |  Epochs: {progress['n_epochs']}")
    print(f"  Started: {progress['start_time']}")
    if progress.get("end_time"):
        print(f"  Ended:   {progress['end_time']}")
    if progress.get("total_time_s"):
        print(f"  Total:   {format_time(progress['total_time_s'])}")
    print(f"  Current: {progress.get('current_model', 'N/A')}")
    print(f"{'='*70}")

    splits = progress.get("splits", {})
    print(f"\n  Data: {splits.get('train_samples', '?')} train / "
          f"{splits.get('val_samples', '?')} val / "
          f"{splits.get('test_samples', '?')} test")

    # Per-model results
    print(f"\n  {'Model':<25} {'Status':<10} {'Epoch':<8} {'Val Recall':<12} {'Test Recall':<12} {'FP/h':<8} {'Time':<8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for model_name in progress.get("models", []):
        mr = progress.get("model_results", {}).get(model_name, {})
        status = mr.get("status", "pending")
        epochs = mr.get("epochs", [])

        if epochs:
            last = epochs[-1]
            epoch_str = f"{last['epoch']}/{progress['n_epochs']}"
            val_recall = f"{last['val']['recall']:.4f}"
            test_recall = f"{last['test']['recall']:.4f}"
            fp_h = f"{last['test']['fp_per_hour']:.2f}"
            total_time = sum(e.get("epoch_time_s", 0) for e in epochs)
            time_str = format_time(total_time)
        else:
            epoch_str = "-"
            val_recall = "-"
            test_recall = "-"
            fp_h = "-"
            time_str = "-"

        # Highlight best
        best_marker = ""
        if mr.get("best_val_recall"):
            best_marker = f" (best={mr['best_val_recall']:.4f}@e{mr.get('best_epoch', '?')})"

        print(f"  {model_name:<25} {status:<10} {epoch_str:<8} {val_recall:<12} "
              f"{test_recall:<12} {fp_h:<8} {time_str:<8}{best_marker}")

    # Detailed epoch history for current/latest model
    current = progress.get("current_model")
    if current and current in progress.get("model_results", {}):
        mr = progress["model_results"][current]
        epochs = mr.get("epochs", [])
        if epochs:
            print(f"\n  Epoch history for {current}:")
            print(f"  {'Ep':<4} {'Loss':<8} {'Val Rec':<9} {'Val F1':<8} {'Test Rec':<9} "
                  f"{'Test F1':<8} {'FP/h':<7} {'SDR':<7} {'Time':<6}")
            print(f"  {'-'*4} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")
            for ep in epochs:
                v = ep["val"]
                t = ep["test"]
                mark = " *" if ep.get("no_improve", 1) == 0 else ""
                print(f"  {ep['epoch']:<4} {ep['train_loss']:<8.4f} {v['recall']:<9.4f} "
                      f"{v['f1']:<8.4f} {t['recall']:<9.4f} {t['f1']:<8.4f} "
                      f"{t['fp_per_hour']:<7.2f} {t['seizure_detection_rate']:<7.3f} "
                      f"{format_time(ep['epoch_time_s']):<6}{mark}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark progress from OBS")
    parser.add_argument("session_id", nargs="?", help="Session ID (timestamp)")
    parser.add_argument("--latest", action="store_true", help="Use most recent session")
    parser.add_argument("--watch", action="store_true", help="Poll every 30s")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    obs = get_obs_client()

    if args.latest or not args.session_id:
        session_id = find_latest_session(obs)
        if not session_id:
            sys.exit(1)
        print(f"Using latest session: {session_id}")
    else:
        session_id = args.session_id

    while True:
        progress = download_progress(obs, session_id)
        if progress is None:
            print(f"No progress.json found for session {session_id}")
            if not args.watch:
                sys.exit(1)
        else:
            if args.json:
                print(json.dumps(progress, indent=2))
            else:
                os.system("clear" if os.name != "nt" else "cls")
                display_progress(progress)
                print(f"  Last checked: {time.strftime('%H:%M:%S')}")

        if not args.watch or (progress and progress.get("status") == "completed"):
            break

        time.sleep(args.interval)

    obs.close()


if __name__ == "__main__":
    main()
