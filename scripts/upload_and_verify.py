"""Upload siena_sop_merged.npz to OBS and verify it is ready for training.

Uploads the merged dataset file and its JSON metadata, then performs a
HEAD / metadata check on OBS to confirm size matches.  Prints the exact
OBS path the training job should mount.

Usage
-----
    # Upload merged NPZ + JSON + SOP data configs:
    python scripts/upload_and_verify.py

    # Check only — confirm already-uploaded file is correct, no re-upload:
    python scripts/upload_and_verify.py --verify-only

    # See what would be uploaded (no transfer):
    python scripts/upload_and_verify.py --dry-run

Requirements
------------
    pip install esdk-obs-python python-dotenv
    Set MODELARTS_AK and MODELARTS_SK in .env or environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _load_env() -> None:
    env_path = _ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass


def _get_obs_client(ak: str, sk: str, endpoint: str):
    try:
        from obs import ObsClient  # type: ignore[import]
    except ImportError:
        print("ERROR: esdk-obs-python not installed.\n"
              "Install with:  pip install esdk-obs-python")
        sys.exit(1)
    server = endpoint if endpoint.startswith("https://") else f"https://{endpoint}"
    return ObsClient(access_key_id=ak, secret_access_key=sk, server=server)


def _parse_obs_uri(uri: str) -> tuple[str, str]:
    """Split 'obs://bucket/path/' → (bucket, 'path/')."""
    assert uri.startswith("obs://"), f"Expected obs:// URI, got {uri!r}"
    parts = uri[6:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/") + "/"


def upload_file(client, bucket: str, key: str, local: Path, dry_run: bool) -> bool:
    size_gb = local.stat().st_size / 1e9
    print(f"  ↑ {local.name}  ({size_gb:.2f} GB)  →  obs://{bucket}/{key}")
    if dry_run:
        print("    [DRY RUN] skipped")
        return True
    resp = client.putFile(bucket, key, str(local))
    if resp.status < 300:
        print(f"    ✓ HTTP {resp.status}")
        return True
    print(f"    ✗ FAILED  HTTP {resp.status}: {resp.errorMessage}")
    return False


def verify_file(client, bucket: str, key: str, expected_bytes: int) -> bool:
    resp = client.getObjectMetadata(bucket, key)
    if resp.status == 200:
        remote_size = int(resp.header.get("content-length", 0)
                          if isinstance(resp.header, dict)
                          else dict(resp.header).get("content-length", 0))
        if remote_size == expected_bytes:
            print(f"    ✓ verified  obs://{bucket}/{key}  ({remote_size/1e9:.2f} GB)")
            return True
        else:
            print(f"    ⚠ size mismatch: local={expected_bytes} remote={remote_size}")
            return False
    elif resp.status == 404:
        print(f"    ✗ NOT FOUND  obs://{bucket}/{key}")
        return False
    else:
        print(f"    ✗ verify error  HTTP {resp.status}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip upload; only verify existing OBS files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without transferring.")
    args = parser.parse_args()

    _load_env()

    # ── Credentials ---------------------------------------------------------
    ak = os.environ.get("MODELARTS_AK") or os.environ.get("MA_AK", "")
    sk = os.environ.get("MODELARTS_SK") or os.environ.get("MA_SK", "")
    region = os.environ.get("MODELARTS_REGION", "cn-north-4")

    if not ak or not sk:
        print("ERROR: MODELARTS_AK / MODELARTS_SK not set.\n"
              "Add them to .env or export as environment variables.")
        sys.exit(1)

    endpoint = f"obs.{region}.myhuaweicloud.com"

    # ── Load backend config --------------------------------------------------
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_ROOT / "configs/backends/modelarts.yaml")
        obs_data_path = cfg.obs_data_path   # obs://auras-experiments/data/processed/
    except Exception:
        obs_data_path = f"obs://auras-experiments/data/processed/"

    bucket, prefix = _parse_obs_uri(obs_data_path)

    # ── Files to upload ------------------------------------------------------
    processed = _ROOT / "data/processed"
    files_to_upload = [
        processed / "siena_sop_merged.npz",
        processed / "siena_sop_merged.json",
    ]

    for f in files_to_upload:
        if not f.exists():
            print(f"ERROR: missing file: {f}\n"
                  f"Run:  python scripts/merge_sop_datasets.py")
            sys.exit(1)

    # ── Summary before action ------------------------------------------------
    total_gb = sum(f.stat().st_size for f in files_to_upload) / 1e9
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}"
          f"{'Verifying' if args.verify_only else 'Uploading'} to OBS")
    print(f"  Endpoint : {endpoint}")
    print(f"  Bucket   : {bucket}")
    print(f"  Prefix   : {prefix}")
    print(f"  Files    : {len(files_to_upload)}  ({total_gb:.2f} GB total)\n")

    client = _get_obs_client(ak, sk, endpoint)
    success = True

    try:
        for f in files_to_upload:
            key = prefix + f.name
            if not args.verify_only:
                ok = upload_file(client, bucket, key, f, dry_run=args.dry_run)
                if not ok:
                    success = False
                    continue
            if not args.dry_run:
                ok = verify_file(client, bucket, key, f.stat().st_size)
                if not ok:
                    success = False
    finally:
        client.close()

    # ── Print training-ready paths ------------------------------------------
    if success:
        print(f"\n{'='*60}")
        print("  Data is ready for training. Use these OBS paths in your job:")
        print()
        print(f"  NPZ  : obs://{bucket}/{prefix}siena_sop_merged.npz")
        print(f"  JSON : obs://{bucket}/{prefix}siena_sop_merged.json")
        print()
        print("  In the ModelArts training job, set:")
        print(f"    data_url = obs://{bucket}/{prefix}")
        print(f"    processed_dir = /cache/data/processed")
        print()
        print("  The training entrypoint will copy the file from OBS to")
        print("  /cache before launching run_experiment.py.")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print("\n✗ One or more files failed. Fix errors above and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
