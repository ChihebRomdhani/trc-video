#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tqdm import tqdm
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings

def augment_data(input_csv, output_csv, target_rows, batch_size,
                 max_cardinality, method):
    df = pd.read_csv(input_csv, encoding='utf-8')
    df = df.reset_index().rename(columns={'index':'id'})

    to_drop = [c for c in df.columns 
               if c != 'id' and df[c].nunique() > max_cardinality]
    if to_drop:
        warnings.warn(f"Dropping {len(to_drop)} "+
                      f"high‑card columns (> {max_cardinality} uniques): {to_drop}")
        df.drop(columns=to_drop, inplace=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.set_primary_key('id')

    if method.lower().startswith('ctgan'):
        synth = CTGANSynthesizer(metadata, verbose=True)
    else:
        synth = GaussianCopulaSynthesizer(metadata)

    print(f"\nFitting {synth.__class__.__name__} on {len(df)} real rows…")
    synth.fit(df)

    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8', errors='ignore') as f:
            existing = sum(1 for _ in f) - 1  # subtract header
    else:
        existing = 0

    if existing <= 0:
        df.iloc[:0].to_csv(output_csv, index=False, encoding='utf-8')

    remaining = target_rows - max(existing, 0)
    if remaining <= 0:
        print(f"Already have {existing} rows ≥ target {target_rows}, nothing to do.")
        return

    print(f"{existing} rows already present, generating {remaining} more…")
    pbar = tqdm(total=remaining, desc="Generating synthetic rows", unit="rows")

    written = 0
    while written < remaining:
        n = min(batch_size, remaining - written)
        new = synth.sample(num_rows=n)
        new.drop(columns=['id'], errors='ignore') \
           .to_csv(output_csv, mode='a', index=False,
                   header=False, encoding='utf-8')
        written += n
        pbar.update(n)

    pbar.close()
    print(f"\nDone — file now has {existing + written} rows total.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Augment a cleaned CSV up to N rows via SDV"
    )
    p.add_argument("--input_csv",   required=True, help="Your cleaned CSV")
    p.add_argument("--output_csv",  required=True, help="Where to write/append augmented CSV")
    p.add_argument("--target_rows", type=int, default=20_000_000,
                   help="Total synthetic rows to generate")
    p.add_argument("--batch_size",  type=int, default=200_000,
                   help="Rows per synth.sample() call")
    p.add_argument("--max_cardinality", type=int, default=100,
                   help="Drop any column with > this many uniques before modeling")
    p.add_argument("--method", choices=["gaussian", "ctgan"], default="gaussian",
                   help="Which SDV synthesizer to use (gaussian=fast, ctgan=deep)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    augment_data(
      input_csv       = args.input_csv,
      output_csv      = args.output_csv,
      target_rows     = args.target_rows,
      batch_size      = args.batch_size,
      max_cardinality = args.max_cardinality,
      method          = args.method
    )
#python DTA8ENT8CALC.py --input_csv processed_data_cleaned.csv --output_csv augmented_20M.csv --target_rows 30000000 --batch_size 200000 --max_cardinality 100 --method gaussian