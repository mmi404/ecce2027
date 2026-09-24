import os
import sys
import time
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/13823687/files/badodd.zip?download=1"
EXPECTED_MD5 = "6a0d3983b379302ab79a99baf7d47d74"
TOTAL_SIZE = 4361793657  # exact bytes

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks
MAX_WORKERS = 4

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
DEST_FILE = os.path.join(RAW_DIR, "badodd.zip")
PART_DIR = os.path.join(RAW_DIR, "badodd_parts")

def compute_md5(filepath):
    print(f"Computing MD5 for {filepath}...")
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_chunk(chunk_id, start_byte, end_byte, max_retries=8):
    part_path = os.path.join(PART_DIR, f"part_{chunk_id:05d}.bin")
    expected_len = end_byte - start_byte + 1
    
    if os.path.exists(part_path) and os.path.getsize(part_path) == expected_len:
        return chunk_id, expected_len, True
        
    headers = {
        "Range": f"bytes={start_byte}-{end_byte}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(ZENODO_URL, headers=headers, timeout=45)
            if r.status_code in (200, 206) and len(r.content) == expected_len:
                with open(part_path + ".tmp", "wb") as f:
                    f.write(r.content)
                os.replace(part_path + ".tmp", part_path)
                return chunk_id, expected_len, False
            else:
                time.sleep(2 * attempt)
        except Exception as e:
            time.sleep(2 * attempt)
            
    raise RuntimeError(f"Failed chunk {chunk_id} ({start_byte}-{end_byte}) after {max_retries} attempts.")

def main():
    if os.path.exists(DEST_FILE):
        if os.path.getsize(DEST_FILE) == TOTAL_SIZE:
            print("Found existing file of exact size. Verifying MD5...")
            if compute_md5(DEST_FILE) == EXPECTED_MD5:
                print("MD5 verified! badodd.zip is ready.")
                return
            else:
                print("MD5 mismatch on existing file. Re-downloading...")
                os.remove(DEST_FILE)
                
    os.makedirs(PART_DIR, exist_ok=True)
    
    # Calculate chunk ranges
    chunks = []
    chunk_id = 0
    start = 0
    while start < TOTAL_SIZE:
        end = min(start + CHUNK_SIZE - 1, TOTAL_SIZE - 1)
        chunks.append((chunk_id, start, end))
        chunk_id += 1
        start = end + 1
        
    total_chunks = len(chunks)
    print(f"Total size: {TOTAL_SIZE / (1024**3):.2f} GB divided into {total_chunks} chunks of {CHUNK_SIZE/(1024*1024):.1f} MB.")
    
    # Check already downloaded chunks
    downloaded_bytes = 0
    needed_chunks = []
    for cid, s, e in chunks:
        part_path = os.path.join(PART_DIR, f"part_{cid:05d}.bin")
        expected_len = e - s + 1
        if os.path.exists(part_path) and os.path.getsize(part_path) == expected_len:
            downloaded_bytes += expected_len
        else:
            needed_chunks.append((cid, s, e))
            
    print(f"Already downloaded: {downloaded_bytes / (1024**3):.2f} GB ({len(chunks) - len(needed_chunks)}/{total_chunks} chunks).")
    
    if needed_chunks:
        with tqdm(total=TOTAL_SIZE, initial=downloaded_bytes, unit="iB", unit_scale=True, desc="Downloading badodd.zip") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(download_chunk, cid, s, e): (cid, e - s + 1)
                    for cid, s, e in needed_chunks
                }
                for future in as_completed(futures):
                    cid, clen, cached = future.result()
                    pbar.update(clen)
                    
    print("\nAll chunks successfully downloaded! Merging into destination file...")
    temp_dest = DEST_FILE + ".assembling"
    with open(temp_dest, "wb") as outfile:
        for cid, s, e in chunks:
            part_path = os.path.join(PART_DIR, f"part_{cid:05d}.bin")
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
                
    print(f"Assembly complete ({os.path.getsize(temp_dest)} bytes). Verifying MD5...")
    actual_md5 = compute_md5(temp_dest)
    if actual_md5 == EXPECTED_MD5:
        os.replace(temp_dest, DEST_FILE)
        print("SUCCESS! badodd.zip verified with MD5:", actual_md5)
        # Clean up parts directory
        for f in os.listdir(PART_DIR):
            os.remove(os.path.join(PART_DIR, f))
        os.rmdir(PART_DIR)
        print("Cleaned up temporary chunk files.")
    else:
        print(f"ERROR: MD5 mismatch! Got {actual_md5}, expected {EXPECTED_MD5}")
        sys.exit(1)

if __name__ == "__main__":
    main()
