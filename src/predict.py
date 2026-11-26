"""Inference helpers for the MILK10K moderate-sampling ResNet18 checkpoint."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
               "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
IMAGE_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def build_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("num_classes", len(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError("Checkpoint class count does not match the 11-class task.")
    if checkpoint.get("class_names") and list(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from CLASS_NAMES.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    with Image.open(image_path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)

def predict(image_path, checkpoint_path="checkpoints/resnet18_moderate_best.pth", model=None, device=None):
    """Return predicted class, confidence, and per-class probabilities."""
    if model is None:
        model, device = build_model(checkpoint_path, device)
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
    }

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_moderate_best.pth")
    args = parser.parse_args()
    print(json.dumps(predict(args.image_path, args.checkpoint), indent=2))
