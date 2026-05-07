"""
Download Scripts
================
Scripts to download raw datasets such as CB513, PDB, etc.

Usage:
  python -m foldnet.data.download --dataset cb513
  python -m foldnet.data.download --dataset cullpdb
  python -m foldnet.data.download --pdb 1A2Y
"""

import os
import urllib.request
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# NOTE: Replace these placeholder URLs with the actual hosting URLs if known.
# Often these are hosted on a lab's server or a Google Drive link.
CB513_URL = "https://raw.githubusercontent.com/example/foldnet-data/main/cb513+profile_split1.npy.gz"
CULLPDB_URL = "https://raw.githubusercontent.com/example/foldnet-data/main/cullpdb+profile_5926_filtered.npy.gz"

def download_file(url: str, dest_path: str):
    """Helper to download a file with basic reporting."""
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path}. Skipping download.")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url}\n -> {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Download complete. ✓")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        logger.error("Please verify the URL is correct and accessible.")

def download_cb513(out_dir: str = "data/raw"):
    """Downloads the CB513 dataset."""
    dest_path = os.path.join(out_dir, "cb513+profile_split1.npy.gz")
    download_file(CB513_URL, dest_path)

def download_cullpdb(out_dir: str = "data/raw"):
    """Downloads the CullPDB training dataset."""
    dest_path = os.path.join(out_dir, "cullpdb+profile_5926_filtered.npy.gz")
    download_file(CULLPDB_URL, dest_path)

def download_pdb(pdb_id: str, out_dir: str = "data/raw/pdb"):
    """
    Downloads a raw PDB file from the RCSB PDB database.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    dest_path = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")
    download_file(url, dest_path)

def main():
    parser = argparse.ArgumentParser(description="Download raw data for FoldNet.")
    parser.add_argument("--dataset", choices=["cb513", "cullpdb", "all"], help="Standard dataset to download")
    parser.add_argument("--pdb", type=str, help="Specific PDB ID to download (e.g., 1A2Y)")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Output directory")
    
    args = parser.parse_args()

    if args.dataset:
        if args.dataset in ["cb513", "all"]:
            download_cb513(args.out_dir)
        if args.dataset in ["cullpdb", "all"]:
            download_cullpdb(args.out_dir)
            
    if args.pdb:
        pdb_out = os.path.join(args.out_dir, "pdb")
        download_pdb(args.pdb, pdb_out)
        
    if not args.dataset and not args.pdb:
        parser.print_help()

if __name__ == "__main__":
    main()
