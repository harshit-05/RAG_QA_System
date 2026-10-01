"""Download the US Army FM instruct dataset (train split, parquet) and preview it.

A standalone helper, not part of the package or the CI suite. Fixed in S1-4 (ISS-18):

* the URL uses Hugging Face's ``/resolve/`` form. The old ``/blob/`` form returns
  the HTML web page (a 128 KB page, saved as if it were the 6.3 MB parquet file);
* the request has a timeout, so a stalled connection cannot hang forever;
* the file is written to ``downloads/`` (gitignored) instead of the current
  directory, and never into ``corpus/``: everything there gets indexed (DEC-4), and
  no loader reads parquet anyway.

Loading the preview needs the ``datasets`` package from the ``eval`` extra::

    uv run --extra eval python scripts/fetch_dataset.py
"""

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
URL = (
    "https://huggingface.co/datasets/Heralax/us-army-fm-instruct/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
TIMEOUT_SECONDS = 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "downloads",
        help="directory to save train.parquet in (default: downloads/, gitignored)",
    )
    args = parser.parse_args(argv)

    out_dir = args.out.resolve()
    if out_dir == CORPUS_DIR or CORPUS_DIR in out_dir.parents:
        print(f"Refusing to write into {CORPUS_DIR}: everything there is indexed (DEC-4).")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / "train.parquet"

    if local_path.exists():
        print(f"{local_path} already exists. Skipping download.")
    else:
        print(f"Downloading {URL}\n        to {local_path} ...")
        try:
            response = requests.get(URL, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return 1
        local_path.write_bytes(response.content)
        print(f"Saved {len(response.content):,} bytes.")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        print("Preview skipped: `datasets` is in the eval extra. Run with `uv run --extra eval`.")
        return 0
    dataset = load_dataset("parquet", data_files={"train": str(local_path)})
    print(dataset)
    print(dataset["train"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
