#!/usr/bin/env python3
"""Minimal embedding extraction using models_vit.py's forward_features directly.

Requires an input CSV with 'Path' and 'Label' columns (Custom3DDataset's
required format). If you don't have real labels, add a dummy 'Label' column
of zeros to the CSV yourself first.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import models_vit
from util.pos_embed import interpolate_pos_embed
from datasets_three_d_fine_resample import Custom3DDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_large_patch16_yo")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--nb_classes", type=int, default=1)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--embedding_col", default="Embedding")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # --- Build model and load checkpoint ---
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        img_size=args.input_size,
    )

    checkpoint = torch.load(args.finetune, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if "pos_embed" in state_dict:
        interpolate_pos_embed(model, state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.finetune}")
    print(msg)

    model.to(device)
    model.eval()

    # --- Dataloading via Custom3DDataset (unmodified) ---
    dataset = Custom3DDataset(csv_path=args.input_csv, resample_size=(args.input_size,) * 3)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    df = pd.read_csv(args.input_csv)
    df[args.embedding_col] = None  # placeholder column, filled in below by row index

    # --- Extract embeddings, writing each batch to its exact source rows ---
    row_idx = 0  # position in the CSV, tracked ourselves since Custom3DDataset returns no index
    with torch.no_grad():
        for volume, _label in loader:
            batch_size = volume.shape[0]
            batch_row_indices = list(range(row_idx, row_idx + batch_size))  # SIDs this batch corresponds to

            volume = volume.to(device)
            embedding = model.forward_features(volume).cpu().numpy()  # [B, embed_dim*2]

            for i, row_i in enumerate(batch_row_indices):
                df.loc[row_i, args.embedding_col] = json.dumps(embedding[i].tolist())

            row_idx += batch_size

    print(f"Wrote {row_idx} embeddings")
    assert row_idx == len(df), (
        f"Mismatch: processed {row_idx} volumes but CSV has {len(df)} rows. "
        "Check for dataset loading errors."
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")


if __name__ == "__main__":
    main()


# Test
# python extract_embeddings_test_model_vit.py --device cuda:1 --input_csv example_inputs/input.csv --finetune checkpoints/tangerine-checkpoint.pth --output_csv example_inputs/output_embeddings_sid_manual_test.csv