"""Check the status of a ModelArts training job.

Usage:
    python scripts/check_job.py <job_id>             # status + metrics
    python scripts/check_job.py <job_id> --events    # include timeline
    python scripts/check_job.py --list               # list recent jobs
"""
import os, sys, requests, json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ak = os.environ['HUAWEI_AK']
sk = os.environ['HUAWEI_SK']
region = os.environ['HUAWEI_REGION']
project_id = os.environ['MODELARTS_PROJECT_ID']

# Auth
iam_url = f'https://iam.{region}.myhuaweicloud.com/v3/auth/tokens'
payload = {'auth': {'identity': {'methods': ['hw_ak_sk'], 'hw_ak_sk': {'access': {'key': ak}, 'secret': {'key': sk}}}, 'scope': {'project': {'id': project_id}}}}
resp = requests.post(iam_url, json=payload, timeout=15)
token = resp.headers['X-Subject-Token']
headers = {'X-Auth-Token': token}
ma_url = f'https://modelarts.{region}.myhuaweicloud.com'


def list_jobs():
    r = requests.get(f'{ma_url}/v2/{project_id}/training-jobs',
                     headers=headers, params={'limit': 10, 'order': 'desc'}, timeout=30)
    jobs = r.json().get('items', [])
    print(f"{'ID':<40} {'Phase':<12} {'Duration':<10} {'Name'}")
    print("-" * 90)
    for j in jobs:
        jid = j['metadata']['id']
        name = j['metadata']['name']
        phase = j['status']['phase']
        dur = j['status'].get('duration', 0) // 1000
        print(f"{jid:<40} {phase:<12} {dur:>6}s    {name}")


def check_job(job_id, show_events=False):
    r = requests.get(f'{ma_url}/v2/{project_id}/training-jobs/{job_id}', headers=headers, timeout=30)
    result = r.json()

    status = result['status']
    phase = status['phase']
    duration = status.get('duration', 0) // 1000
    spec = result.get('spec', {}).get('resource', {})
    flavor = spec.get('flavor_id', '?')
    flavor_info = spec.get('flavor_detail', {}).get('flavor_info', {})

    print(f"Job:      {job_id}")
    print(f"Phase:    {phase}")
    print(f"Duration: {duration}s ({duration//60}m {duration%60}s)")
    print(f"Flavor:   {flavor}")
    if flavor_info:
        cpu = flavor_info.get('cpu', {})
        mem = flavor_info.get('memory', {})
        gpu = flavor_info.get('gpu', {})
        disk = flavor_info.get('disk', {})
        print(f"          CPU: {cpu.get('core_num','?')} cores | RAM: {mem.get('size','?')} GiB | "
              f"GPU: {gpu.get('unit_num','0')}x {gpu.get('product_name','')} {gpu.get('memory','')} | "
              f"Disk: {disk.get('size','?')} GB")

    # Metrics
    metrics = status.get('metrics_statistics')
    if metrics:
        print(f"\n--- Resource Metrics (avg/max) ---")
        cpu_m = metrics.get('cpu_usage', {})
        mem_m = metrics.get('mem_usage', {})
        gpu_m = metrics.get('gpu', {})
        print(f"  CPU:      {cpu_m.get('average',0):.1f}% / {cpu_m.get('max',0):.1f}%")
        print(f"  RAM:      {mem_m.get('average',0):.1f}% / {mem_m.get('max',0):.1f}%")
        if gpu_m:
            print(f"  GPU util: {gpu_m.get('util',{}).get('average',0):.1f}% / {gpu_m.get('util',{}).get('max',0):.1f}%")
            print(f"  GPU mem:  {gpu_m.get('mem_usage',{}).get('average',0):.1f}% / {gpu_m.get('mem_usage',{}).get('max',0):.1f}%")

    # Events timeline
    if show_events or phase not in ('Completed', 'Running'):
        r2 = requests.get(f'{ma_url}/v2/{project_id}/training-jobs/{job_id}/events',
                         headers=headers, params={'order': 'asc', 'limit': 50}, timeout=15)
        if r2.status_code == 200:
            events = r2.json().get('events', [])
            print(f"\n--- Timeline ({len(events)} events) ---")
            for e in events:
                t = e['time'][11:19]
                print(f"  {t}  [{e['source']:4s}] {e['message']}")


if __name__ == '__main__':
    if '--list' in sys.argv:
        list_jobs()
    else:
        job_id = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
        if not job_id:
            print("Usage: python scripts/check_job.py <job_id> [--events] [--list]")
            sys.exit(1)
        show_events = '--events' in sys.argv
        check_job(job_id, show_events)
