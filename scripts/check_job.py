"""Check, list, or terminate cloud training jobs.

Usage:
    python scripts/check_job.py <job_id>
    python scripts/check_job.py <job_id> --terminate
    python scripts/check_job.py --provider aws <job_id>
    python scripts/check_job.py --list
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.cloud import get_provider
from scripts.cloud.base import JobPhase


def display_status(status):
    """Pretty-print job status."""
    print(f"Job:      {status.job_id}")
    print(f"Phase:    {status.phase.value}")
    print(f"Duration: {status.duration_s}s ({status.duration_s // 60}m {status.duration_s % 60}s)")
    if status.flavor:
        print(f"Flavor:   {status.flavor}")
    if status.flavor_detail:
        print(f"          {status.flavor_detail}")

    if status.metrics:
        m = status.metrics
        print(f"\n--- Resource Metrics (avg/max) ---")
        print(f"  CPU:      {m.cpu_avg:.1f}% / {m.cpu_max:.1f}%")
        print(f"  RAM:      {m.ram_avg:.1f}% / {m.ram_max:.1f}%")
        if m.gpu_util_avg >= 0:
            print(f"  GPU util: {m.gpu_util_avg:.1f}% / {m.gpu_util_max:.1f}%")
            print(f"  GPU mem:  {m.gpu_mem_avg:.1f}% / {m.gpu_mem_max:.1f}%")

    if status.events:
        print(f"\n--- Timeline ({len(status.events)} events) ---")
        for e in status.events:
            t = e.get("time", "")
            if len(t) > 11:
                t = t[11:19]
            src = e.get("source", "?")
            msg = e.get("message", "")
            print(f"  {t}  [{src:4s}] {msg}")


def list_jobs(provider):
    """List recent jobs."""
    jobs = provider.training().list_jobs(limit=10)
    print(f"{'ID':<45} {'Phase':<12} {'Duration':<10} {'Name'}")
    print("-" * 95)
    for j in jobs:
        dur = f"{j.duration_s}s"
        print(f"{j.job_id:<45} {j.phase.value:<12} {dur:<10} {j.flavor}")


def main():
    parser = argparse.ArgumentParser(description="Check cloud training job status")
    parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("CLOUD_PROVIDER", "huawei"),
        choices=["huawei", "aws"],
    )
    parser.add_argument("job_id", nargs="?", help="Job ID to check")
    parser.add_argument("--list", action="store_true", help="List recent jobs")
    parser.add_argument("--terminate", action="store_true", help="Terminate the job")
    parser.add_argument("--events", action="store_true", help="Show event timeline")
    args = parser.parse_args()

    provider = get_provider(args.provider)

    if args.list:
        list_jobs(provider)
    elif args.job_id:
        if args.terminate:
            provider.training().terminate(args.job_id)
            print(f"Terminate requested for {args.job_id}")
        else:
            status = provider.training().get_status(args.job_id)
            display_status(status)
    else:
        parser.print_help()
        sys.exit(1)

    provider.close()


if __name__ == "__main__":
    main()
