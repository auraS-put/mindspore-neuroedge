"""Submit a training job to the cloud (Huawei ModelArts or AWS SageMaker).

Usage:
    python scripts/submit_job.py --provider huawei --name test
    python scripts/submit_job.py --provider aws --name test --flavor ml.p3.2xlarge
    python scripts/submit_job.py --benchmark --env RUN_MODE=full EPOCHS=5
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.cloud import get_provider, JobConfig


DEFAULT_FLAVORS = {
    "huawei": "modelarts.vm.gpu.v100",
    "aws": "ml.p3.2xlarge",
}

DEFAULT_DATA_PATHS = {
    "huawei": "/auras-experiments/data/",
    "aws": "s3://auras-experiments/data/",
}

DEFAULT_OUTPUT_PATHS = {
    "huawei": "/auras-experiments/output/",
    "aws": "s3://auras-experiments/output/",
}


def build_code_tarball() -> str:
    """Build a tarball of the source code and configs."""
    tarball_path = os.path.join(tempfile.gettempdir(), "auras_code.tar.gz")
    with tarfile.open(tarball_path, "w:gz") as tf:
        tf.add("src/auras", arcname="auras")
        tf.add("configs", arcname="configs")
        tf.add("scripts", arcname="scripts")
    return tarball_path


def upload_code(provider, boot_script: str) -> None:
    """Upload boot script and code tarball to cloud storage."""
    store = provider.storage()

    with open(boot_script, "r") as f:
        boot_content = f.read()
    store.upload_bytes(boot_content.encode(), "code/boot.py")
    print(f"Uploaded boot.py ({boot_script})")

    tarball_path = build_code_tarball()
    size_kb = os.path.getsize(tarball_path) / 1024
    store.upload_file(tarball_path, "code/auras_code.tar.gz")
    print(f"Uploaded tarball ({size_kb:.0f} KB)")
    os.remove(tarball_path)


def main():
    parser = argparse.ArgumentParser(description="Submit a cloud training job")
    parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("CLOUD_PROVIDER", "huawei"),
        choices=["huawei", "aws"],
        help="Cloud provider (default: $CLOUD_PROVIDER or huawei)",
    )
    parser.add_argument("--name", default="auras", help="Job name prefix")
    parser.add_argument("--skip-upload", action="store_true", help="Skip code upload")
    parser.add_argument("--flavor", default=None, help="Instance type")
    parser.add_argument("--data-path", default=None, help="Remote data path")
    parser.add_argument("--output-path", default=None, help="Remote output path")
    parser.add_argument(
        "--env", nargs="*", default=[],
        help="Extra env vars: KEY=VALUE",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Use benchmark boot script (multi-model)",
    )
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    provider = get_provider(args.provider)
    flavor = args.flavor or DEFAULT_FLAVORS[args.provider]
    data_path = args.data_path or DEFAULT_DATA_PATHS[args.provider]
    output_path = args.output_path or DEFAULT_OUTPUT_PATHS[args.provider]

    boot_script = (
        "scripts/cloud_boot_benchmark.py" if args.benchmark
        else "scripts/cloud_boot.py"
    )

    if not args.skip_upload:
        upload_code(provider, boot_script)

    env_vars = {}
    for kv in args.env:
        k, v = kv.split("=", 1)
        env_vars[k] = v

    # Provider-specific internal config
    if args.provider == "huawei":
        bucket = os.environ.get("HUAWEI_OBS_BUCKET", "auras-experiments")
        env_vars["__code_dir"] = f"/{bucket}/code/"
        env_vars["__boot_file"] = f"/{bucket}/code/boot.py"

    config = JobConfig(
        name=args.name,
        boot_script=boot_script,
        code_paths=["src/auras", "configs", "scripts"],
        data_path=data_path,
        output_path=output_path,
        flavor=flavor,
        env_vars=env_vars,
    )

    job_id = provider.training().submit_job(config)
    print(f"\nJob submitted successfully!")
    print(f"  Provider: {provider.name}")
    print(f"  ID:       {job_id}")
    print(f"  Flavor:   {flavor}")
    print(f"\nCheck status with:")
    print(f"  python scripts/check_job.py --provider {args.provider} {job_id}")

    provider.close()


if __name__ == "__main__":
    main()
