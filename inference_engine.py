# -*- coding: utf-8 -*-
"""
Core Inference, Face Cropping, and Results Storage Engine for Student Engagement Recognition.
Handles:
- Arbitrary image dimensions, resolutions, and orientations.
- Robust face detection with MediaPipe Face Landmarker & padding square crop (Lanczos sinc resampling).
- Live 52 FACS blendshape extraction.
- Multi-model inference (Dual-Branch ResNet/EfficientNet, Image-Only ResNet/EfficientNet).
- Automatic archiving into 'Testing_Results/<test_session>/' with images, CSV, JSON, and summary reports.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "DL_models"))
sys.path.append(os.path.join(BASE_DIR, "Feature_extract"))

from DL_models.dual_branch_network import get_upgraded_model
from DL_models.multimodal_dataloader import BLENDSHAPE_COLUMNS
from Feature_extract.mediapipe_facs_extractor import get_face_landmarker, BLENDSHAPE_NAMES
import mediapipe as mp

TESTING_RESULTS_DIR = os.path.join(BASE_DIR, "Testing_Results")
os.makedirs(TESTING_RESULTS_DIR, exist_ok=True)

CLASS_LABELS = {
    0: "Disengaged",
    1: "Nominally Engaged",
    2: "Highly Engaged"
}

CLASS_DESCRIPTIONS = {
    0: "Disengaged (Inattentive / Looking Away / Slumped / Bored)",
    1: "Nominally Engaged (Attentive / Neutral / Listening to Screen)",
    2: "Highly Engaged (Actively Participating / Responsive / Focused)"
}

CLASS_COLORS = {
    0: (239, 68, 68),    # Red / Orange
    1: (59, 130, 246),   # Blue
    2: (16, 185, 129)    # Green
}

MODEL_CONFIGS = {
    "Dual-Branch ResNet-18 (Recommended)": {
        "backbone": "resnet",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_resnet.pth"),
        "description": "Visual facial crops (224x224) fused with 52 MediaPipe FACS muscle blendshapes."
    },
    "Multimodal Ensemble (ResNet-18 + EfficientNet + FACS)": {
        "is_ensemble": True,
        "models": ["Dual-Branch ResNet-18 (Recommended)", "Dual-Branch EfficientNet-B0"],
        "weights": [0.55, 0.45],
        "description": "Fuses ResNet-18 and EfficientNet-B0 predictions with 52 FACS blendshapes for peak accuracy (69.68% / 71.91% F1)."
    },
    "Dual-Branch ResNet-34": {
        "backbone": "resnet34",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_resnet34.pth"),
        "description": "34-layer residual visual features fused with 52 MediaPipe FACS blendshapes."
    },
    "Dual-Branch ResNet-50": {
        "backbone": "resnet50",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_resnet50.pth"),
        "description": "50-layer Bottleneck residual features (2,048 dims) fused with 52 MediaPipe FACS blendshapes."
    },
    "Dual-Branch EfficientNet-B0": {
        "backbone": "efficientnet",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_efficientnet.pth"),
        "description": "Mobile inverted bottleneck features fused with 52 MediaPipe FACS blendshapes."
    },
    "Dual-Branch EfficientNet-B1": {
        "backbone": "efficientnet_b1",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_efficientnet_b1.pth"),
        "description": "Compound scaled EfficientNet-B1 features fused with 52 MediaPipe FACS blendshapes."
    },
    "Dual-Branch MobileNetV3": {
        "backbone": "mobilenet_v3",
        "image_only": False,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_upgraded_mobilenet_v3.pth"),
        "description": "Lightweight MobileNetV3 visual features fused with 52 MediaPipe FACS blendshapes."
    },
    "Image-Only ResNet-18": {
        "backbone": "resnet",
        "image_only": True,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_image_only_resnet.pth"),
        "description": "Visual features only (224x224) without MediaPipe blendshapes."
    },
    "Image-Only EfficientNet-B0": {
        "backbone": "efficientnet",
        "image_only": True,
        "checkpoint": os.path.join(BASE_DIR, "DL_models", "hyper-1", "best_model_image_only_efficientnet.pth"),
        "description": "EfficientNet visual features only (224x224) without MediaPipe blendshapes."
    }
}

_LOADED_MODELS = {}

def get_loaded_model(model_key="Dual-Branch ResNet-18 (Recommended)", device=None):
    """
    Caches and returns the requested PyTorch model in evaluation mode.
    """
    global _LOADED_MODELS
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key], device

    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["Dual-Branch ResNet-18 (Recommended)"])
    ckpt_path = config["checkpoint"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}. Please train the model first.")

    model = get_upgraded_model(
        backbone_type=config["backbone"],
        num_classes=3,
        image_only=config["image_only"]
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _LOADED_MODELS[model_key] = model
    return model, device


def crop_face_and_extract_blendshapes(pil_image, target_size=(224, 224), margin=0.20):
    """
    Takes an input PIL Image of ANY dimension or resolution:
    1. Normalizes orientation via EXIF.
    2. Runs MediaPipe FaceLandmarker to detect face landmarks and 52 FACS blendshapes.
    3. Crops face with square aspect ratio and padding margin.
    4. Resamples with anti-aliased Lanczos sinc interpolation to target_size (224x224).
    5. Returns: (cropped_pil, annotated_pil, blendshapes_np, face_detected, bbox_coords)
    """
    # 1. Normalize EXIF orientation
    pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')
    orig_w, orig_h = pil_image.size
    img_np = np.array(pil_image)

    detector = get_face_landmarker()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    detection_result = detector.detect(mp_img)

    blendshapes_np = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
    face_detected = False
    bbox = None

    # Check if landmarks and blendshapes were detected
    if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
        face_detected = True
        landmarks = detection_result.face_landmarks[0]
        
        # Bounding box calculation in pixels
        xs = [lm.x * orig_w for lm in landmarks]
        ys = [lm.y * orig_h for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        face_w = max_x - min_x
        face_h = max_y - min_y
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Square bounding box with margin
        box_size = max(face_w, face_h) * (1.0 + margin)
        
        x1 = int(cx - box_size / 2.0)
        y1 = int(cy - box_size / 2.0)
        x2 = int(cx + box_size / 2.0)
        y2 = int(cy + box_size / 2.0)

        bbox = (max(0, int(min_x)), max(0, int(min_y)), min(orig_w, int(max_x)), min(orig_h, int(max_y)))

        # Handle crop with boundary padding if necessary
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - orig_w)
        pad_bottom = max(0, y2 - orig_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_np = cv2.copyMakeBorder(
                img_np, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT_101
            )
            crop_x1 = x1 + pad_left
            crop_y1 = y1 + pad_top
            crop_x2 = x2 + pad_left
            crop_y2 = y2 + pad_top
            face_crop_np = padded_np[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            face_crop_np = img_np[y1:y2, x1:x2]

        cropped_pil = Image.fromarray(face_crop_np).resize(target_size, Image.Resampling.LANCZOS)

        # Extract 52 FACS Blendshapes
        if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
            b_list = detection_result.face_blendshapes[0]
            name_to_idx = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
            for item in b_list:
                if item.category_name in name_to_idx:
                    blendshapes_np[name_to_idx[item.category_name]] = item.score
                elif item.index < len(blendshapes_np):
                    blendshapes_np[item.index] = item.score

    else:
        # Fallback: Central square crop if face landmarks are missing
        min_dim = min(orig_w, orig_h)
        cx, cy = orig_w // 2, orig_h // 2
        x1 = cx - min_dim // 2
        y1 = cy - min_dim // 2
        cropped_pil = pil_image.crop((x1, y1, x1 + min_dim, y1 + min_dim)).resize(target_size, Image.Resampling.LANCZOS)
        bbox = (x1, y1, x1 + min_dim, y1 + min_dim)

    # Generate annotated image
    annotated_pil = pil_image.copy()
    draw = ImageDraw.Draw(annotated_pil)
    if bbox is not None and face_detected:
        bx1, by1, bx2, by2 = bbox
        # Draw green bounding box around detected face
        line_w = max(2, int(min(orig_w, orig_h) * 0.005))
        draw.rectangle([bx1, by1, bx2, by2], outline="#10b981", width=line_w)

    return cropped_pil, annotated_pil, blendshapes_np, face_detected, bbox


def predict_engagement_level(cropped_pil, blendshapes_np, model_key="Dual-Branch ResNet-18 (Recommended)"):
    """
    Performs inference on the 224x224 cropped face and blendshapes.
    Returns: (predicted_class, class_name, confidence_percent, probabilities_dict, top_facs_list)
    """
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["Dual-Branch ResNet-18 (Recommended)"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if config.get("is_ensemble"):
        probs_accum = np.zeros(3, dtype=np.float32)
        blend_np = blendshapes_np if blendshapes_np is not None else np.zeros(52, dtype=np.float32)
        for sub_model_key, w in zip(config["models"], config["weights"]):
            sub_model, dev = get_loaded_model(sub_model_key)
            sub_cfg = MODEL_CONFIGS[sub_model_key]
            img_tensor = transform(cropped_pil).unsqueeze(0).to(dev)
            if sub_cfg["image_only"]:
                with torch.no_grad():
                    logits = sub_model(img_tensor)
                    p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                blend_tensor = torch.from_numpy(blend_np).unsqueeze(0).to(dev)
                with torch.no_grad():
                    logits = sub_model(img_tensor, blend_tensor)
                    p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_accum += w * p
        probs = probs_accum
    else:
        model, device = get_loaded_model(model_key)
        img_tensor = transform(cropped_pil).unsqueeze(0).to(device)
        if config["image_only"]:
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            blend_tensor = torch.from_numpy(blendshapes_np).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(img_tensor, blend_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class] * 100.0)

    probs_dict = {
        "Disengaged (0)": float(probs[0]),
        "Nominally Engaged (1)": float(probs[1]),
        "Highly Engaged (2)": float(probs[2])
    }

    # Top 5 most active facial muscle actions
    top_indices = np.argsort(blendshapes_np)[::-1][:5]
    top_facs = [
        {"name": BLENDSHAPE_NAMES[idx], "score": float(blendshapes_np[idx])}
        for idx in top_indices if blendshapes_np[idx] > 0.01
    ]
    if len(top_facs) == 0:
        top_facs = [{"name": "neutral", "score": float(blendshapes_np[0])}]

    return predicted_class, CLASS_LABELS[predicted_class], confidence, probs_dict, top_facs


def save_test_session(original_pil, cropped_pil, annotated_pil, blendshapes_np,
                      pred_class, class_label, confidence, probs_dict, top_facs,
                      model_key, filename="upload.jpg"):
    """
    Stores the uploaded image, cropped face, blendshapes CSV, and prediction report
    into a dedicated test directory: Testing_Results/test_<timestamp>_<clean_filename>/
    """
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
    folder_name = f"test_{timestamp_str}_{clean_name}"
    session_dir = os.path.join(TESTING_RESULTS_DIR, folder_name)
    os.makedirs(session_dir, exist_ok=True)

    # 1. Save original image
    orig_path = os.path.join(session_dir, f"original_{clean_name}.jpg")
    original_pil.save(orig_path, format="JPEG", quality=95)

    # 2. Save cropped face (224x224)
    crop_path = os.path.join(session_dir, "cropped_face_224.jpg")
    cropped_pil.save(crop_path, format="JPEG", quality=95)

    # 3. Save annotated image with bounding box
    annotated_path = os.path.join(session_dir, "annotated_detection.jpg")
    annotated_pil.save(annotated_path, format="JPEG", quality=95)

    # 4. Save 52 MediaPipe FACS blendshapes CSV
    df_blendshapes = pd.DataFrame([{
        BLENDSHAPE_NAMES[i]: blendshapes_np[i] for i in range(len(BLENDSHAPE_NAMES))
    }])
    facs_csv_path = os.path.join(session_dir, "facs_blendshapes.csv")
    df_blendshapes.to_csv(facs_csv_path, index=False)

    # 5. Save structured JSON result
    result_data = {
        "filename": filename,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_dimensions": list(original_pil.size),
        "model_used": model_key,
        "predicted_class": pred_class,
        "predicted_label": class_label,
        "confidence_percentage": round(confidence, 2),
        "class_probabilities": {k: round(v, 4) for k, v in probs_dict.items()},
        "top_active_facs": top_facs,
        "session_directory": session_dir
    }
    json_path = os.path.join(session_dir, "prediction_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # 6. Save human-readable summary report
    report_path = os.path.join(session_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"STUDENT ENGAGEMENT PREDICTION REPORT\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date & Time:         {result_data['timestamp']}\n")
        f.write(f"Original File:       {filename} ({original_pil.size[0]}x{original_pil.size[1]}px)\n")
        f.write(f"Model Evaluated:     {model_key}\n")
        f.write(f"Predicted State:     {class_label} (Class {pred_class})\n")
        f.write(f"Confidence:          {confidence:.2f}%\n")
        f.write(f"\nClass Probabilities:\n")
        for k, v in probs_dict.items():
            f.write(f"  - {k}: {v*100:.2f}%\n")
        f.write(f"\nTop Active Facial Muscle Blendshapes:\n")
        for facs in top_facs:
            f.write(f"  - {facs['name']}: {facs['score']:.4f}\n")
        f.write(f"{'='*50}\n")

    # 7. Append to global history log: Testing_Results/all_test_history.csv
    history_csv = os.path.join(TESTING_RESULTS_DIR, "all_test_history.csv")
    history_entry = {
        "timestamp": result_data["timestamp"],
        "folder": folder_name,
        "filename": filename,
        "model": model_key,
        "predicted_class": pred_class,
        "predicted_label": class_label,
        "confidence": f"{confidence:.1f}%",
        "prob_disengaged": f"{probs_dict['Disengaged (0)']*100:.1f}%",
        "prob_nominal": f"{probs_dict['Nominally Engaged (1)']*100:.1f}%",
        "prob_high": f"{probs_dict['Highly Engaged (2)']*100:.1f}%"
    }
    df_hist_entry = pd.DataFrame([history_entry])
    if os.path.exists(history_csv):
        df_hist = pd.read_csv(history_csv)
        df_hist = pd.concat([df_hist, df_hist_entry], ignore_index=True)
    else:
        df_hist = df_hist_entry
    df_hist.to_csv(history_csv, index=False)

    return session_dir, result_data
