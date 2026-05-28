"""Submit a training job to ModelArts.

Usage:
    python scripts/submit_job.py              # submit with defaults
    python scripts/submit_job.py --name test  # custom job name suffix
"""
import os, sys, argparse, time, json, tarfile, requests
from obs import ObsClient, HeadPermission
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ak = os.environ['HUAWEI_AK']
sk = os.environ['HUAWEI_SK']
region = os.environ['HUAWEI_REGION']
project_id = os.environ['MODELARTS_PROJECT_ID']
bucket = os.environ.get('HUAWEI_OBS_BUCKET', 'auras-experiments')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='auras', help='Job name prefix')
parser.add_argument('--skip-upload', action='store_true', help='Skip code/boot upload')
parser.add_argument('--flavor', default='modelarts.vm.gpu.v100')
parser.add_argument('--data-path', default='/auras-experiments/data_test/', help='OBS data input path')
parser.add_argument('--env', nargs='*', default=[], help='Extra env vars: KEY=VALUE (e.g. RUN_MODE=full EPOCHS=1)')
parser.add_argument('--benchmark', action='store_true', help='Use benchmark boot script (multi-model)')
args = parser.parse_args()


def upload_code():
    """Upload boot script + code tarball to OBS."""
    obs_client = ObsClient(access_key_id=ak, secret_access_key=sk,
                           server=f'https://obs.{region}.myhuaweicloud.com')

    # Upload boot script (choose benchmark or standard)
    boot_script = 'scripts/cloud_boot_benchmark.py' if args.benchmark else 'scripts/cloud_boot.py'
    with open(boot_script, 'r') as f:
        boot_content = f.read()
    resp = obs_client.putContent(bucket, 'code/boot.py', content=boot_content)
    print(f"Upload boot.py ({boot_script}): {resp.status}")

    # Build and upload code tarball
    tarball_path = '/tmp/auras_code.tar.gz'
    with tarfile.open(tarball_path, 'w:gz') as tf:
        tf.add('src/auras', arcname='auras')
        tf.add('configs', arcname='configs')
        tf.add('scripts', arcname='scripts')
    size_kb = os.path.getsize(tarball_path) / 1024
    resp = obs_client.putFile(bucket, 'code/auras_code.tar.gz', tarball_path)
    print(f"Upload tarball ({size_kb:.0f} KB): {resp.status}")

    obs_client.close()


def submit_job():
    """Submit training job via ModelArts API."""
    iam_url = f'https://iam.{region}.myhuaweicloud.com/v3/auth/tokens'
    payload = {'auth': {'identity': {'methods': ['hw_ak_sk'], 'hw_ak_sk': {'access': {'key': ak}, 'secret': {'key': sk}}}, 'scope': {'project': {'id': project_id}}}}
    resp = requests.post(iam_url, json=payload, timeout=15)
    token = resp.headers['X-Subject-Token']
    headers = {'X-Auth-Token': token, 'Content-Type': 'application/json'}

    ma_url = f'https://modelarts.{region}.myhuaweicloud.com'
    job_name = f'{args.name}-{int(time.time())}'

    # Build environment variables list
    env_vars = {}
    for kv in args.env:
        k, v = kv.split('=', 1)
        env_vars[k] = v

    algorithm = {
        'code_dir': f'/{bucket}/code/',
        'boot_file': f'/{bucket}/code/boot.py',
        'engine': {'engine_id': 'mindspore_1.3.0-cuda_10.1-py_3.7-ubuntu_1804-x86_64'},
        'inputs': [{'name': 'data', 'remote': {'obs': {'obs_url': args.data_path}}}],
        'outputs': [{'name': 'output', 'remote': {'obs': {'obs_url': f'/{bucket}/output/'}}}]
    }
    if env_vars:
        algorithm['environments'] = env_vars

    job_body = {
        'kind': 'job',
        'metadata': {'name': job_name, 'description': 'Training job'},
        'algorithm': algorithm,
        'spec': {'resource': {'flavor_id': args.flavor, 'node_count': 1}}
    }

    r = requests.post(f'{ma_url}/v2/{project_id}/training-jobs', headers=headers, json=job_body, timeout=30)
    result = r.json()

    if r.status_code == 201:
        job_id = result['metadata']['id']
        print(f"\nJob submitted successfully!")
        print(f"  Name:   {job_name}")
        print(f"  ID:     {job_id}")
        print(f"  Status: {result['status']['phase']}")
        print(f"\nCheck status with:")
        print(f"  python scripts/check_job.py {job_id}")
    else:
        print(f"Submit failed: {r.status_code}")
        print(json.dumps(result, indent=2)[:800])


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    if not args.skip_upload:
        upload_code()

    submit_job()
