"""Upload processed data files to Huawei OBS.

Transfers the processed .npz file(s) to the OBS bucket configured in
configs/backends/modelarts.yaml so that ModelArts training jobs can access
them without re-generating data in the cloud.

Usage
-----
  # Upload all SOP variant npz files using .env credentials:
  python scripts/upload_obs.py

  # Upload a specific file:
  python scripts/upload_obs.py --file data/processed/siena_sop15.npz

  # Upload to a custom OBS prefix:
  python scripts/upload_obs.py --obs-prefix data/processed/

  # Dry-run (print what would be uploaded, no actual transfer):
  python scripts/upload_obs.py --dry-run

Requirements
------------
  pip install esdk-obs-python  (Huawei OBS Python SDK)
  Credentials via .env or environment variables:
    MODELARTS_AK, MODELARTS_SK, MODELARTS_REGION
  OBS endpoint / bucket from configs/backends/modelarts.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _load_env():
    """Load .env file if present."""
    env_path = _ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)


def _load_backend_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(_ROOT / "configs/backends/modelarts.yaml")


def _parse_bucket_key(obs_uri: str) -> tuple[str, str]:
    """Split 'obs://bucket/path/key' into (bucket, key)."""
    if not obs_uri.startswith("obs://"):
        raise ValueError(f"Expected obs:// URI, got: {obs_uri!r}")
    parts = obs_uri[6:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def upload_file(
    local_path: Path,
    bucket: str,
    obs_key: str,
    ak: str,
    sk: str,
    endpoint: str,
    *,
    dry_run: bool = False,
) -> None:
    """Upload a single file to OBS."""
    size_gb = local_path.stat().st_size / 1e9
    print(f"  {local_path.name}  ({size_gb:.2f} GB)  →  obs://{bucket}/{obs_key}")

    if dry_run:
        print("  [DRY RUN] skipping actual upload")
        return

    try:
        from obs import ObsClient  # type: ignore[import]
    except ImportError:
        print(
            "\nERROR: esdk-obs-python not installed.\n"
            "Install with:  pip install esdk-obs-python\n"
            "Then re-run this script."
        )
        sys.exit(1)

    server = endpoint if endpoint.startswith("https://") else f"https://{endpoint}"
    client = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)

    try:
        resp = client.putFile(bucket, obs_key, str(local_path))
        if resp.status < 300:
            print(f"  ✓ uploaded  (HTTP {resp.status})")
        else:
            print(f"  ✗ upload failed  (HTTP {resp.status}): {resp.errorMessage}")
            sys.exit(1)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload processed EEG data to Huawei OBS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Local .npz file to upload. Defaults to uploading all .npz files in data/processed/.",
    )
    parser.add_argument(
        "--obs-prefix",
        default=None,
        help="OBS key prefix (folder) to upload into. Defaults to obs_data_path in modelarts.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without transferring.",
    )
    args = parser.parse_args()

    _load_env()
    cfg = _load_backend_cfg()

    # --- Resolve credentials ------------------------------------------------
    ak = os.environ.get("MODELARTS_AK") or os.environ.get("MA_AK") or cfg.get("ak", "")
    sk = os.environ.get("MODELARTS_SK") or os.environ.get("MA_SK") or cfg.get("sk", "")
    region = os.environ.get("MODELARTS_REGION", cfg.get("region", "cn-north-4"))

    if not ak or not sk:
        print(
            "ERROR: OBS credentials not found.\n"
            "Set MODELARTS_AK and MODELARTS_SK in your .env file or environment."
        )
        sys.exit(1)

    endpoint = f"obs.{region}.myhuaweicloud.com"

    # --- Resolve OBS destination --------------------------------------------
    obs_data_path = args.obs_prefix or cfg.obs_data_path  # e.g. obs://auras-experiments/data/processed/
    bucket, prefix = _parse_bucket_key(obs_data_path)
    if not prefix.endswith("/"):
        prefix += "/"

    # --- Resolve local file(s) to upload ------------------------------------
    if args.file:
        files = [Path(args.file)]
    else:
        processed_dir = _ROOT / "data/processed"
        files = list(processed_dir.glob("*.npz"))
        if not files:
            print(f"ERROR: No .npz files found in {processed_dir}")
            sys.exit(1)

    # Validate files exist
    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: file not found: {f}")
        sys.exit(1)

    # --- Upload -------------------------------------------------------------
    print(f"\nOBS endpoint : {endpoint}")
    print(f"Bucket       : {bucket}")
    print(f"Prefix       : {prefix}")
    print(f"Files        : {len(files)}\n")

    for local_path in files:
        obs_key = prefix + local_path.name
        upload_file(local_path, bucket, obs_key, ak, sk, endpoint, dry_run=args.dry_run)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
