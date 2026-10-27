"""
01 — Download, Filter, and Clean the Corpus
============================================

Step 1 — download every pretrain shard from HuggingFace (~5 GB, cached).
Step 2 — measure whether Type-A cleaning still earns its place.
Step 3 — demo the Type-B normalizer (unchanged, still the golden rule).
Step 4 — write a SAMPLED corpus for the tokenizer to train on.

"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from Cleaning import clean_dataframe, normalize_text, prepare_document
from huggingface_hub import hf_hub_download, snapshot_download 

REPO = "bakrianoo/jabarti-llm-dataset"
MAX_CHUNKS = 20


HF_FILES = {
"phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
"phase1_eval": "pretrain/phase1_eval-00000-of-00001.parquet",

"phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
"phase2_eval": "pretrain/phase2_eval-00000-of-00001.parquet",

"ft_train": "finetune/train-00000-of-00001.parquet",
"ft_eval": "finetune/eval-00000-of-00001.parquet",
}
 
TRAIN_SPLITS = {"phase1_train", "phase2_train"}
EVAL_SPLITS = {"phase1_eval", "phase2_eval"}
OUT_DIR = Path(__file__).parent / "output"

def download_and_filter():
    """Download raw HF splits, apply hard cap to train splits, write *_filtered.parquet."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)   # check if the directory is exist 
    for name, filename in HF_FILES.items():
        print(f"Load: {name}")
        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=filename,
        )
        df = pd.read_parquet(local) 

        if name in TRAIN_SPLITS:
            before_filter = len(df)

            df = (
                df.sort_values(["article_id", "chunk_index"])
                  .groupby("article_id", sort=False)
                  .head(MAX_CHUNKS)
                  .reset_index(drop=True)
            )
            after_filter = len(df) 
            print(f"df: {name} => {before_filter} filtered to be {after_filter}") 

        output_path = OUT_DIR / f"{name}_filtered.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")


def clean_phase(phase, input_path, output_path):
    print(f"==== clean_phase: {phase} ====")

    df = pd.read_parquet(input_path)

    print(f"  Source        : {input_path}")
    print(f"  Rows          : {len(df):,}")

    cleaned_df = clean_dataframe(df)

    removed_rows = len(df) - len(cleaned_df)
    print(f"  Rows kept        : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)


def main():
    download_and_filter()
    for name in TRAIN_SPLITS:
        input_path = OUT_DIR / f"{name}_filtered.parquet"
        output_path = OUT_DIR / f"{name}_clean.parquet"

        clean_phase(phase= name , input_path=input_path , output_path= output_path) 

    for name in EVAL_SPLITS:
            input_path = OUT_DIR / f"{name}_filtered.parquet"
            output_path = OUT_DIR / f"{name}_clean.parquet"
    
            clean_phase(phase= name , input_path=input_path , output_path= output_path) 
    

if __name__ =="__main__" :
    main()
        


