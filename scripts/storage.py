"""Download outputs or manage cloud storage (OBS / S3).

Usage:
    python scripts/storage.py download                      # download all outputs
    python scripts/storage.py download --prefix output/benchmark_20260526_072421/
    python scripts/storage.py ls                            # list top-level prefixes
    python scripts/storage.py ls output/                    # list objects under prefix
    python scripts/storage.py --provider aws download
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.cloud import get_provider


def cmd_ls(store, prefix: str, recursive: bool) -> None:
    """List objects or prefixes in cloud storage."""
    if recursive:
        objects = store.list_objects(prefix)
        for key in objects:
            if not key.endswith("/"):
                print(f"  {key}")
    else:
        # Show subdirectories
        prefixes = store.list_prefixes(prefix)
        if prefixes:
            for p in sorted(prefixes):
                print(f"  {p}")
        # Show files at this level
        objects = store.list_objects(prefix, delimiter="/")
        for key in objects:
            if not key.endswith("/") and key != prefix:
                print(f"  {key}")


def cmd_download(store, prefix: str, local_dir: str, skip: list[str]) -> None:
    """Download objects from cloud storage to local directory."""
    objects = store.list_objects(prefix)

    # Filter out directory markers and skipped prefixes
    to_download = []
    for key in objects:
        if key.endswith("/"):
            continue
        if any(key.startswith(s) for s in skip):
            continue
        to_download.append(key)

    print(f"Found {len(to_download)} objects to download")
    downloaded = 0
    failed = 0

    for key in to_download:
        local_path = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            data = store.get_bytes(key)
            with open(local_path, "wb") as f:
                f.write(data)
            downloaded += 1
            size_str = f"{len(data) / 1024:.1f} KB" if len(data) > 1024 else f"{len(data)} B"
            print(f"  OK  {key} ({size_str})")
        except Exception as e:
            failed += 1
            err_msg = str(e)
            # Truncate long error messages
            if len(err_msg) > 80:
                err_msg = err_msg[:80] + "..."
            print(f"  FAIL {key}: {err_msg}")

    print(f"\nDone: {downloaded} downloaded, {failed} failed")
    if failed and "arrear" in str(e).lower():
        print("\nHint: Account has insufficient balance. Top up to enable downloads.")


def main():
    parser = argparse.ArgumentParser(description="Cloud storage operations")
    parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("CLOUD_PROVIDER", "huawei"),
        choices=["huawei", "aws"],
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ls
    ls_parser = subparsers.add_parser("ls", help="List storage contents")
    ls_parser.add_argument("prefix", nargs="?", default="", help="Prefix to list")
    ls_parser.add_argument("-r", "--recursive", action="store_true", help="List recursively")

    # download
    dl_parser = subparsers.add_parser("download", help="Download from storage")
    dl_parser.add_argument(
        "--prefix", default="output/",
        help="Remote prefix to download (default: output/)",
    )
    dl_parser.add_argument(
        "--local-dir", default="experiments/obs_output",
        help="Local directory to save files",
    )
    dl_parser.add_argument(
        "--skip", nargs="*", default=["data/", "data_test/", "wheels/"],
        help="Prefixes to skip",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    provider = get_provider(args.provider)
    store = provider.storage()

    if args.command == "ls":
        cmd_ls(store, args.prefix, args.recursive)
    elif args.command == "download":
        cmd_download(store, args.prefix, args.local_dir, args.skip)

    provider.close()


if __name__ == "__main__":
    main()
