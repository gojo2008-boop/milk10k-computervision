"""Generate an 11-class MILK10k-style submission CSV from image metadata."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from predict import build_model, preprocess_image, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="CSV containing lesion_id, isic_id and image_type")
    parser.add_argument("--image-dir", required=True, help="Directory containing ISIC image files")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_moderate_best.pth")
    parser.add_argument("--output", default="submission_generated.csv")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    required = {"lesion_id", "isic_id", "image_type"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    derm = metadata[metadata["image_type"].astype(str).str.lower().str.contains("dermoscopic", na=False)].copy()
    if derm.empty:
        raise ValueError("No dermoscopic rows found in metadata.")
    if derm["lesion_id"].duplicated().any():
        raise ValueError("Metadata has more than one dermoscopic row per lesion.")
    model, device = build_model(args.checkpoint)
    rows = []
    image_dir = Path(args.image_dir)
    for i, row in enumerate(derm.itertuples(index=False), start=1):
        image_path = image_dir / f"{row.isic_id}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{row.isic_id}.jpeg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {row.isic_id}: expected .jpg/.jpeg in {image_dir}")
        x = preprocess_image(image_path).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        record = {"lesion_id": row.lesion_id}
        record.update({name: float(probs[j]) for j, name in enumerate(CLASS_NAMES)})
        rows.append(record)
        if i % 50 == 0:
            print(f"Processed {i}/{len(der)}")
    output = pd.DataFrame(rows, columns=["lesion_id"] + CLASS_NAMES)
    if output["lesion_id"].duplicated().any() or len(output) != len(derm):
        raise RuntimeError("Submission lesion uniqueness/row-count check failed.")
    if not np.allclose(output[CLASS_NAMES].sum(axis=1).to_numpy(), 1.0, atol=1e-4):
        raise RuntimeError("Probability rows do not sum to one.")
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} rows to {args.output}")

if __name__ == "__main__":
    main()
