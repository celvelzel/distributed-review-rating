#!/usr/bin/env python3
"""Diagnose and fix Kaggle submission upload."""
import os, json, sys, time, requests, io

os.environ['KAGGLE_USERNAME'] = 'rickyma1028'
os.environ['KAGGLE_KEY'] = 'KGAT_9895fa87525d5a9a3514ae8bd156320b'

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    json.dump({'username': 'rickyma1028', 'key': 'KGAT_9895fa87525d5a9a3514ae8bd156320b'}, f)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

filepath = 'output/archive/submissions/sub-20260622-01-ve65-rlg35.csv'

# Step 1: Get the upload URL
with api.build_kaggle_client() as kaggle:
    from kagglesdk.competitions.services.competition_api_service import ApiStartSubmissionUploadRequest
    request = ApiStartSubmissionUploadRequest()
    request.competition_name = 'comp-5434-2526-sem-3-project'
    request.file_name = os.path.basename(filepath)
    request.content_length = os.path.getsize(filepath)
    request.last_modified_epoch_seconds = int(os.path.getmtime(filepath))
    
    response = kaggle.competitions.competition_api_client.start_submission_upload(request)
    upload_url = response.create_url
    token = response.token
    print(f'Upload URL: {upload_url}', flush=True)
    url_host = upload_url.split('/')[2]
    print(f'URL host: {url_host}', flush=True)

# Step 2: Try direct PUT with timeout
print(f'\nTrying PUT with 30s timeout...', flush=True)
file_size = os.path.getsize(filepath)
with open(filepath, 'rb') as f:
    start = time.time()
    try:
        session = requests.Session()
        session.headers.update({
            'Content-Length': str(file_size),
        })
        resp = session.put(upload_url, data=f, timeout=(10, 30))
        print(f'PUT result: {resp.status_code} in {time.time()-start:.1f}s', flush=True)
        print(f'Response: {resp.text[:200]}', flush=True)
    except requests.Timeout as e:
        print(f'PUT TIMEOUT after {time.time()-start:.1f}s: {e}', flush=True)
    except requests.ConnectionError as e:
        print(f'PUT CONNECTION ERROR after {time.time()-start:.1f}s: {e}', flush=True)
    except Exception as e:
        print(f'PUT ERROR after {time.time()-start:.1f}s: {type(e).__name__}: {e}', flush=True)

# Step 3: Test connectivity to the upload host
print(f'\nTesting connectivity to {url_host}...', flush=True)
import socket
try:
    ip = socket.gethostbyname(url_host)
    print(f'DNS: {url_host} -> {ip}', flush=True)
except Exception as e:
    print(f'DNS error: {e}', flush=True)

try:
    sock = socket.create_connection((url_host, 443), timeout=10)
    print(f'TCP connection OK', flush=True)
    sock.close()
except Exception as e:
    print(f'TCP error: {e}', flush=True)
