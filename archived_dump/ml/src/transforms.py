"""Train/eval transforms. eval_transforms and preprocess_spec MUST stay in sync:
the backend re-implements eval preprocessing from preprocess.json without torch."""
from torchvision import transforms as T


def train_transforms(cfg: dict) -> T.Compose:
    im = cfg["image"]
    return T.Compose([
        T.RandomResizedCrop(im["size"], scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(im["mean"], im["std"]),
    ])


def eval_transforms(cfg: dict) -> T.Compose:
    im = cfg["image"]
    return T.Compose([
        T.Resize(im["resize"], interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(im["size"]),
        T.ToTensor(),
        T.Normalize(im["mean"], im["std"]),
    ])


def preprocess_spec(cfg: dict) -> dict:
    """Written to preprocess.json next to model.onnx."""
    im = cfg["image"]
    return {
        "resize_shorter": im["resize"],
        "crop_size": im["size"],
        "interpolation": im["interpolation"],
        "mean": im["mean"],
        "std": im["std"],
        "color": "RGB",
        "layout": "NCHW",
        "scale": "0-1",      # pixels divided by 255 before mean/std
        "output": "logits",  # apply softmax to get probabilities
    }
