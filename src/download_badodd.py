import os
import sys
import hashlib
import requests
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/13823687/files/badodd.zip?download=1"
EXPECTED_MD5 = "6a0d3983b379302ab79a99baf7d47d74"
EXPECTED_SIZE = 4361793657  # ~4.36 GB

DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
DEST_FILE = os.path.join(DEST_DIR, "badodd.zip")

def compute_md5(filepath):
    print(f"Computing MD5 for {filepath}...")
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"
    
    resume_header = {}
    downloaded_bytes = 0
    if os.path.exists(temp_path):
        downloaded_bytes = os.path.getsize(temp_path)
        resume_header = {"Range": f"bytes={downloaded_bytes}-"}
        print(f"Resuming download from byte {downloaded_bytes}...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    headers.update(resume_header)
    
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # Range not satisfiable, file might be complete
            print("Range not satisfiable; resetting temp file...")
            downloaded_bytes = 0
            temp_path = dest_path + ".part"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            headers.pop("Range", None)
            r = requests.get(url, headers=headers, stream=True, timeout=60)
        
        r.raise_for_status()
        
        total_size = int(r.headers.get("content-length", 0)) + downloaded_bytes
        mode = "ab" if downloaded_bytes > 0 and r.status_code == 206 else "wb"
        if mode == "wb":
            downloaded_bytes = 0
            
        with open(temp_path, mode) as f, tqdm(
            desc="Downloading badodd.zip",
            initial=downloaded_bytes,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 2):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"Downloaded successfully to {dest_path}")

def main():
    if os.path.exists(DEST_FILE):
        print(f"Found existing {DEST_FILE} (size: {os.path.getsize(DEST_FILE)} bytes). Checking MD5...")
        md5_val = compute_md5(DEST_FILE)
        if md5_val == EXPECTED_MD5:
            print("MD5 matches expected checksum! Skipping download.")
            return
        else:
            print(f"MD5 mismatch: got {md5_val}, expected {EXPECTED_MD5}. Re-downloading...")
            os.remove(DEST_FILE)
            
    print(f"Downloading BadODD dataset from Zenodo to {DEST_FILE}...")
    download_file(ZENODO_URL, DEST_FILE)
    
    print("Verifying downloaded file checksum...")
    actual_md5 = compute_md5(DEST_FILE)
    if actual_md5 == EXPECTED_MD5:
        print("Verification SUCCESS: MD5 checksum matches!")
    else:
        print(f"Verification FAILURE: Expected {EXPECTED_MD5}, got {actual_md5}")
        sys.exit(1)

if __name__ == "__main__":
    main()
