"""
SmartCrop AI - PlantDoc Deployment Verification Script
Verifies presence, integrity, and operational inference readiness of PlantDoc
disease classification models, configurations, and evaluation artifacts.
"""

import sys
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("verify_deployment")

# Import central project configuration
try:
    from src.config.plantdoc_config import (
        PROJECT_ROOT,
        DISEASE_MODEL_PATH,
        DISEASE_CLASSES_PATH,
        DISEASE_METRICS_PATH,
        BASELINE_MODEL_PATH,
        COMPARISON_RESULTS_PATH,
        NUM_CLASSES,
        IMAGE_SIZE,
        DROPOUT_RATE,
        CLASSIFIER_HIDDEN_DIM,
        DEVICE,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DISEASE_MODEL_PATH = PROJECT_ROOT / "models" / "disease" / "disease_model.pth"
    DISEASE_CLASSES_PATH = PROJECT_ROOT / "models" / "disease" / "disease_classes.json"
    DISEASE_METRICS_PATH = PROJECT_ROOT / "models" / "disease" / "disease_metrics.json"
    BASELINE_MODEL_PATH = PROJECT_ROOT / "outputs" / "baseline_disease_model.pth"
    COMPARISON_RESULTS_PATH = PROJECT_ROOT / "outputs" / "comparison_results.json"
    NUM_CLASSES = 27
    IMAGE_SIZE = (224, 224)
    DROPOUT_RATE = 0.4
    CLASSIFIER_HIDDEN_DIM = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Model Architectures
# ==============================================================================

class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(scale))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


def replace_bn_with_in(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            setattr(module, name, nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False))
        else:
            replace_bn_with_in(child)
    return module


class PaperInspiredResNet50(nn.Module):
    """
    Model A (Baseline): ResNet-50 with Instance Normalization and CBAM attention.
    Replaces BatchNorm with InstanceNorm (affine=True) and appends CBAM after layer4.
    """
    def __init__(self, num_classes: int = 27, dropout: float = 0.4):
        super().__init__()
        base = models.resnet50(weights=None)
        replace_bn_with_in(base)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_proposed_model(num_classes: int = 27, dropout: float = 0.4) -> nn.Module:
    """
    Model B (Proposed): EfficientNet-B0 with custom classifier head.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )
    return model


# ==============================================================================
# Verification Logic
# ==============================================================================

def verify_file_presence() -> dict:
    """Verify presence and file sizes of all required artifacts."""
    logger.info("--> Step 1: Checking required file artifacts...")
    artifacts = {
        "disease_model_pth": DISEASE_MODEL_PATH,
        "disease_classes_json": DISEASE_CLASSES_PATH,
        "disease_metrics_json": DISEASE_METRICS_PATH,
        "baseline_model_pth": BASELINE_MODEL_PATH,
        "comparison_results_json": COMPARISON_RESULTS_PATH,
    }

    status = {}
    all_present = True
    for key, path in artifacts.items():
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        status[key] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2)
        }
        if exists:
            logger.info(f"  [OK] {key}: {path} ({status[key]['size_mb']} MB)")
        else:
            logger.error(f"  [MISSING] {key}: {path}")
            all_present = False

    if not all_present:
        raise FileNotFoundError("One or more required deployment files are missing.")
    return status


def verify_metadata_and_metrics() -> dict:
    """Validate JSON metadata, class list, and metric consistency."""
    logger.info("--> Step 2: Validating metadata and metrics JSON files...")

    # 1. Disease classes
    with open(DISEASE_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)

    if isinstance(classes_data, list):
        class_list = classes_data
    elif isinstance(classes_data, dict):
        class_list = [classes_data[str(i)] if str(i) in classes_data else classes_data[i] for i in range(len(classes_data))]
    else:
        raise ValueError("disease_classes.json format not recognized (must be list or dict).")

    num_classes = len(class_list)
    logger.info(f"  [OK] Verified {num_classes} classes in disease_classes.json.")
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {num_classes}.")

    # 2. Disease metrics (Proposed Model B)
    with open(DISEASE_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_b = json.load(f)

    metrics_dict = metrics_b.get("metrics", metrics_b)
    acc = metrics_dict.get("accuracy", metrics_dict.get("test_accuracy", 0.0))
    f1 = metrics_dict.get("macro_f1", metrics_dict.get("test_macro_f1", 0.0))

    logger.info(f"  [OK] Proposed Model B Metrics: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

    # 3. Comparison results
    with open(COMPARISON_RESULTS_PATH, "r", encoding="utf-8") as f:
        comp_results = json.load(f)

    logger.info("  [OK] Comparison results JSON successfully parsed.")

    return {
        "num_classes": num_classes,
        "classes": class_list,
        "metrics_proposed": metrics_b,
        "comparison_results": comp_results
    }


def verify_proposed_model_loading_and_inference(classes: list) -> dict:
    """Verify Model B (EfficientNet-B0) loading, architecture, and dummy forward pass."""
    logger.info("--> Step 3: Verifying Model B (Proposed EfficientNet-B0)...")

    model = build_proposed_model(num_classes=len(classes), dropout=DROPOUT_RATE)
    checkpoint = torch.load(DISEASE_MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        val_f1 = checkpoint.get("val_f1", None)
        val_acc = checkpoint.get("val_acc", None)
        logger.info(f"  [OK] Loaded checkpoint dict (epoch={epoch}, val_f1={val_f1}, val_acc={val_acc})")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    logger.info("  [OK] Weights loaded successfully with strict=True.")

    # Dummy inference test
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device=DEVICE)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, len(classes)), f"Expected logits shape (1, {len(classes)}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"
    assert not torch.isinf(logits).any(), "Logits contain Inf values!"

    probs = torch.softmax(logits, dim=1)
    prob_sum = float(probs.sum().item())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1: {prob_sum}"

    pred_idx = int(torch.argmax(probs, dim=1).item())
    assert 0 <= pred_idx < len(classes), f"Predicted class index {pred_idx} out of range [0, {len(classes)-1}]"
    pred_label = classes[pred_idx]

    logger.info(f"  [OK] Forward pass successful. Shape: {tuple(logits.shape)}, Predicted class: '{pred_label}' (idx: {pred_idx}).")
    return {
        "status": "PASS",
        "output_shape": list(logits.shape),
        "dummy_pred_class": pred_label,
        "dummy_pred_idx": pred_idx,
        "device_used": DEVICE
    }


def verify_baseline_model_loading_and_inference(classes: list) -> dict:
    """Verify Model A (Paper-Inspired ResNet-50+IN+CBAM) loading and dummy forward pass."""
    logger.info("--> Step 4: Verifying Model A (Baseline ResNet-50 + IN + CBAM)...")

    model = PaperInspiredResNet50(num_classes=len(classes), dropout=DROPOUT_RATE)
    checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        val_f1 = checkpoint.get("val_f1", None)
        val_acc = checkpoint.get("val_acc", None)
        logger.info(f"  [OK] Loaded baseline checkpoint dict (epoch={epoch}, val_f1={val_f1}, val_acc={val_acc})")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    logger.info("  [OK] Baseline weights loaded successfully with strict=True.")

    # Dummy inference test
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device=DEVICE)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (1, len(classes)), f"Expected logits shape (1, {len(classes)}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"
    assert not torch.isinf(logits).any(), "Logits contain Inf values!"

    pred_idx = int(torch.argmax(logits, dim=1).item())
    assert 0 <= pred_idx < len(classes), f"Predicted class index {pred_idx} out of range [0, {len(classes)-1}]"
    pred_label = classes[pred_idx]

    logger.info(f"  [OK] Baseline forward pass successful. Shape: {tuple(logits.shape)}, Predicted class: '{pred_label}' (idx: {pred_idx}).")
    return {
        "status": "PASS",
        "output_shape": list(logits.shape),
        "dummy_pred_class": pred_label,
        "dummy_pred_idx": pred_idx,
        "device_used": DEVICE
    }


def run_all_verifications() -> dict:
    """Run full verification pipeline and return audit summary."""
    logger.info("================================================================================")
    logger.info("         SMARTCROP AI - PLANTDOC DEPLOYMENT VERIFICATION PIPELINE               ")
    logger.info("================================================================================")

    files_status = verify_file_presence()
    meta_status = verify_metadata_and_metrics()
    model_b_status = verify_proposed_model_loading_and_inference(meta_status["classes"])
    model_a_status = verify_baseline_model_loading_and_inference(meta_status["classes"])

    logger.info("================================================================================")
    logger.info("                  [SUCCESS] ALL VERIFICATIONS PASSED                            ")
    logger.info("================================================================================")

    return {
        "verification_result": "SUCCESS",
        "artifacts": files_status,
        "metadata": {
            "num_classes": meta_status["num_classes"],
            "classes": meta_status["classes"]
        },
        "model_proposed_verification": model_b_status,
        "model_baseline_verification": model_a_status
    }


if __name__ == "__main__":
    try:
        report = run_all_verifications()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Verification FAILED with error: {e}")
        sys.exit(1)
