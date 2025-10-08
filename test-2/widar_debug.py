#!/usr/bin/env python3
"""
widar_debug.py

Debug helper: recursively find .dat files under a folder (case-insensitive),
print a sample list, show first bytes of the first file, and attempt
to load it (tries text CSV/whitespace and raw float32 fallback).

Usage:
    python widar_debug.py --data-dir /data/widar3.0/raw
"""

import os, argparse, binascii
from glob import glob
import numpy as np

def find_dat_files_recursive(data_dir, exts=('.dat', '.DAT')):
    matches = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            for ext in exts:
                if fname.endswith(ext):
                    matches.append(os.path.join(root, fname))
                    break
    return sorted(matches)

def try_load_text(path):
    try:
        arr = np.loadtxt(path)
        return ('loadtxt', arr)
    except Exception as e:
        return ('loadtxt_error', e)

def try_genfromtxt(path):
    try:
        arr = np.genfromtxt(path, delimiter=',')
        return ('genfromtxt', arr)
    except Exception as e:
        return ('genfromtxt_error', e)

def try_fromfile_float32(path):
    try:
        arr = np.fromfile(path, dtype=np.float32)
        return ('fromfile_float32', arr)
    except Exception as e:
        return ('fromfile_error', e)

def hexdump_head(path, nbytes=256):
    with open(path, 'rb') as f:
        head = f.read(nbytes)
    try:
        # also try to decode as utf-8 for readable text preview
        text = head.decode('utf-8', errors='replace')
    except:
        text = None
    hexs = binascii.hexlify(head).decode('ascii')
    return text, hexs, len(head)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/data/widar3.0/raw')
    parser.add_argument('--show-n', type=int, default=20, help='how many filenames to show')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"[Error] directory does not exist: {data_dir}")
        return

    files = find_dat_files_recursive(data_dir)
    print(f"[Info] Found {len(files)} files with .dat/.DAT under {data_dir}")
    if len(files) == 0:
        # show a few filenames in the tree to help debug
        print("[Info] Directory listing (top-level):")
        for i, e in enumerate(sorted(os.listdir(data_dir))[:100]):
            print("  ", e)
        return

    print("\nSample files:")
    for f in files[:args.show_n]:
        print("  ", f)
    sample = files[0]
    print("\n[Sample file] ", sample)
    st = os.stat(sample)
    print(" Size (bytes):", st.st_size)

    text, hexs, n = hexdump_head(sample, nbytes=512)
    print("\n--- First 512 bytes as text (may contain non-printable):")
    print(text)
    print("\n--- First 512 bytes as hex (truncated):")
    print(hexs[:512])

    # attempt to load with multiple strategies
    print("\nAttempting to load with numpy.loadtxt...")
    res = try_load_text(sample)
    if res[0] == 'loadtxt':
        arr = res[1]
        print(" loadtxt succeeded. shape:", getattr(arr, 'shape', None), "dtype:", getattr(arr, 'dtype', None))
        print(" first 5 rows:\n", arr[:5])
        return
    else:
        print(" loadtxt failed:", res[1])

    print("\nAttempting to load with numpy.genfromtxt (comma-delimited)...")
    res = try_genfromtxt(sample)
    if res[0] == 'genfromtxt':
        arr = res[1]
        print(" genfromtxt succeeded. shape:", getattr(arr, 'shape', None), "dtype:", getattr(arr, 'dtype', None))
        print(" first 5 rows:\n", arr[:5])
        return
    else:
        print(" genfromtxt failed:", res[1])

    print("\nAttempting to read raw as float32 with np.fromfile...")
    res = try_fromfile_float32(sample)
    if res[0] == 'fromfile_float32':
        arr = res[1]
        print(" fromfile float32 succeeded. length:", arr.size)
        # try to guess 2D shape if divisible by common widths (e.g., 30, 90)
        for w in (90, 30, 6, 3, 2):
            if arr.size % w == 0:
                print(f" possible reshape to (-1, {w}) -> shape {(arr.size // w, w)}")
        print(" first 20 values:\n", arr[:20])
        return
    else:
        print(" fromfile failed:", res[1])

    print("\nAll load attempts failed. If files are in a custom binary format (e.g. complex CSI),\n"
          "please paste the first 512 bytes output above or a small example file and I will adapt the loader.")

if __name__ == '__main__':
    main()
