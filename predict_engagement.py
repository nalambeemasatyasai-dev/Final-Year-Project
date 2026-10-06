# -*- coding: utf-8 -*-
"""
Inference and Verification Tool for Student Engagement Recognition.
Features:
- Random Unseen Testing: Randomly samples N unseen test images and validates >= 80% accuracy (e.g., 8/10 correct).
- Single Image Live Inference: Takes any custom facial photo, extracts live 52 FACS MediaPipe blendshapes,
  applies Lanczos resampling, and predicts engagement level with full confidence breakdown.
"""

import os
import sys
import argparse
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "DL_models"))
sys.path.append(os.path.join(SCRIPT_DIR, "Feature_extract"))

from DL_models.dual_branch_network import get_upgraded_model
from DL_models.multimodal_dataloader import BLENDSHAPE_COLUMNS
from Feature_extract.mediapipe_facs_extractor import extract_blendshapes_from_image, get_face_landmarker

CLASS_LABELS = {
    0: "Disengaged (Bored / Inattentive / Looking Away)",
    1: "Nominally Engaged (Attentive / Neutral / Listening)",
    2: "Highly Engaged (Actively Participating / Responsive)"
}

CLASS_SHORT = {
    0: "Disengaged (0)",
    1: "Nominally Engaged (1)",
    2: "Highly Engaged (2)"
}

def load_trained_model(checkpoint_path=None, device='cpu', preferred_backbone='resnet'):
    if checkpoint_path is None:
        if preferred_backbone == 'resnet':
            candidate_paths = [
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded_resnet.pth"),
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded_efficientnet.pth"),
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded.pth"),
            ]
        else:
            candidate_paths = [
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded_efficientnet.pth"),
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded_resnet.pth"),
                os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "best_model_upgraded.pth"),
            ]
        for p in candidate_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break
                
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get('backbone', preferred_backbone)
    img_size = ckpt.get('img_size', 224)

    model = get_upgraded_model(backbone_type=backbone, num_classes=3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model [{backbone.upper()}] from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model, img_size, backbone


def predict_single_image(model, img_path, device, img_size=224, blendshapes=None):
    """
    Runs dual-branch inference on an image.
    Uses Lanczos resampling and live 52 FACS MediaPipe blendshapes.
    """
    # 1. Image preprocessing with Lanczos resampling (zero information loss)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 2. Extract or use supplied blendshapes
    if blendshapes is None:
        detector = get_face_landmarker()
        blendshapes_np = extract_blendshapes_from_image(img_path, detector=detector)
    else:
        blendshapes_np = np.array(blendshapes, dtype=np.float32)

    blendshapes_tensor = torch.from_numpy(blendshapes_np).unsqueeze(0).to(device)

    # 3. Model forward pass
    with torch.no_grad():
        logits = model(img_tensor, blendshapes_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))

    return predicted_class, probabilities, blendshapes_np


def test_random_unseen_samples(model, num_samples=10, device='cpu', img_size=224, seed=None):
    """
    Randomly samples N images from the unseen test split and evaluates accuracy.
    Verifies that >= 80% (e.g. 8/10) are correctly classified.
    """
    test_split_path = os.path.join(SCRIPT_DIR, "DL_models", "hyper-1", "test_split_unseen.pkl")
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split not found at {test_split_path}. Run training first!")

    test_df = pd.read_pickle(test_split_path)
    print(f"\n=======================================================")
    print(f"  RANDOM UNSEEN SAMPLE VERIFICATION (Testing {num_samples} Samples)")
    print(f"  Pool size: {len(test_df)} unseen test images")
    print(f"=======================================================\n")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Sample balanced or uniform random unseen samples
    sample_indices = random.sample(range(len(test_df)), min(num_samples, len(test_df)))
    sampled_rows = test_df.iloc[sample_indices].copy()

    correct_count = 0
    results_table = []

    print(f"{'#':<4} | {'Filename':<18} | {'True Ground Truth':<25} | {'Model Prediction':<25} | {'Confidence':<10} | {'Status'}")
    print("-" * 105)

    for i, (_, row) in enumerate(sampled_rows.iterrows(), 1):
        img_path = row['image_path']
        true_label = int(row['class_label'])
        filename = row.get('filename', os.path.basename(img_path))

        # Extract blendshapes from the cached dataframe row
        blendshape_vals = row[BLENDSHAPE_COLUMNS].values.astype(np.float32)

        pred_class, probs, _ = predict_single_image(
            model=model,
            img_path=img_path,
            device=device,
            img_size=img_size,
            blendshapes=blendshape_vals
        )

        is_correct = (pred_class == true_label)
        if is_correct:
            correct_count += 1
            status_str = "CORRECT (v)"
        else:
            status_str = "WRONG   (x)"

        conf_str = f"{probs[pred_class] * 100:.1f}%"
        true_str = CLASS_SHORT[true_label]
        pred_str = CLASS_SHORT[pred_class]

        print(f"{i:<4} | {filename:<18} | {true_str:<25} | {pred_str:<25} | {conf_str:<10} | {status_str}")

        results_table.append({
            'sample_num': i,
            'filename': filename,
            'true_label': true_label,
            'pred_label': pred_class,
            'confidence': probs[pred_class],
            'correct': is_correct
        })

    accuracy = (correct_count / len(sampled_rows)) * 100.0
    print("-" * 105)
    print(f"\nFinal Score: {correct_count} / {len(sampled_rows)} Correct ({accuracy:.1f}%)")
    
    target_met = (correct_count / len(sampled_rows)) >= 0.80
    if target_met:
        print(f"SUCCESS: Target achieved! (>= 80% accuracy on random unseen test data)\n")
    else:
        print(f"Score was below 80%. Target was not met on this specific batch.\n")

    return accuracy


class EngagementPredictor:
    def __init__(self, checkpoint_path=None, backbone='resnet', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model, self.img_size, self.backbone = load_trained_model(checkpoint_path, self.device, preferred_backbone=backbone)

    def predict(self, img_path, blendshapes=None):
        return predict_single_image(self.model, img_path, self.device, self.img_size, blendshapes)

    def evaluate_random_unseen_samples(self, n_samples=10, seed=42):
        return test_random_unseen_samples(self.model, num_samples=n_samples, device=self.device, img_size=self.img_size, seed=seed)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Engagement Inference and Testing")
    parser.add_argument("--test_random", type=int, default=10, help="Number of random unseen samples to test (e.g. 10)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to classify")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, img_size, backbone = load_trained_model(args.checkpoint, device=device)

    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
        pred, probs, blendshapes = predict_single_image(model, args.image, device=device, img_size=img_size)
        print(f"\nImage: {args.image}")
        print(f"Predicted Engagement Level: {CLASS_LABELS[pred]}")
        print(f"Class Probabilities:")
        for c in range(3):
            print(f"  - {CLASS_SHORT[c]}: {probs[c]*100:.2f}%")
        
        # Display top 5 active facial blendshapes
        top_indices = np.argsort(blendshapes)[::-1][:5]
        print(f"\nTop Active Facial Muscle Blendshapes (FACS):")
        for idx in top_indices:
            print(f"  - {BLENDSHAPE_COLUMNS[idx]}: {blendshapes[idx]:.3f}")
    else:
        test_random_unseen_samples(
            model=model,
            num_samples=args.test_random,
            device=device,
            img_size=img_size,
            seed=args.seed
        )
