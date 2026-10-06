# -*- coding: utf-8 -*-
"""
Upgraded MediaPipe Face Blendshape Extractor for Student Engagement Detection.
Extracts 52 FACS (Facial Action Coding System) muscle activation scores
using Google MediaPipe Face Landmarker (face_landmarker.task).

Features extracted:
- 52 standard FACS action units (brow furrow, eye blink, mouth smile, jaw drop, etc.)
- Values are bounded [0.0, 1.0], scale-invariant, illumination-robust, and rotation-aware.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# 52 Standard MediaPipe Face Blendshape Names
BLENDSHAPE_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft',
    'noseSneerRight'
]

_GLOBAL_DETECTOR = None

def get_face_landmarker(model_path=None):
    """
    Initializes and returns a singleton MediaPipe FaceLandmarker instance.
    """
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is not None:
        return _GLOBAL_DETECTOR
    
    if model_path is None:
        # Search relative to current script or repo root
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task"),
            os.path.join(os.path.dirname(__file__), "face_landmarker.task"),
            "face_landmarker.task",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_path = os.path.abspath(p)
                break
                
    if model_path is None or not os.path.exists(model_path):
        target_path = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")
        target_path = os.path.abspath(target_path)
        print("[INFO] Downloading face_landmarker.task from Google MediaPipe CDN...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, target_path)
        model_path = target_path

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3
    )
    _GLOBAL_DETECTOR = vision.FaceLandmarker.create_from_options(options)
    return _GLOBAL_DETECTOR

def extract_blendshapes_from_image(image_input, detector=None):
    """
    Extracts 52 FACS blendshapes from an image file path or numpy array (RGB).
    Returns a numpy array of shape (52,) with float32 values [0.0, 1.0].
    """
    if detector is None:
        detector = get_face_landmarker()

    if isinstance(image_input, str):
        # Read from file using PIL to preserve exact pixel values
        img_pil = Image.open(image_input).convert('RGB')
        img_rgb = np.array(img_pil, dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 1:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image_input
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert('RGB'), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    # Initialize zero vector
    features = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)

    if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
        blendshapes = detection_result.face_blendshapes[0]
        # Map by category name or index
        name_to_idx = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
        for category in blendshapes:
            if category.category_name in name_to_idx:
                features[name_to_idx[category.category_name]] = category.score
            elif category.index < len(features):
                features[category.index] = category.score

    return features

def extract_dataset_blendshapes(data_dir, output_csv=None, output_pkl=None):
    """
    Extracts 52 blendshapes for all images in WACV data (folders '0', '1', '2').
    Saves to CSV and Pickle for fast lookup.
    """
    detector = get_face_landmarker()
    records = []
    
    classes = ['0', '1', '2']
    print(f"Starting MediaPipe 52-Blendshape extraction from {data_dir}...")
    
    total_count = sum(len(os.listdir(os.path.join(data_dir, c))) for c in classes if os.path.exists(os.path.join(data_dir, c)))
    processed = 0

    for label in classes:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            continue
        
        file_list = sorted(os.listdir(folder_path))
        for filename in file_list:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, filename)
            img_id = os.path.splitext(filename)[0]
            
            try:
                blendshapes = extract_blendshapes_from_image(img_path, detector=detector)
            except Exception as e:
                print(f"Warning: Failed to extract features for {filename}: {e}")
                blendshapes = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)

            row = {
                'image_id': img_id,
                'filename': filename,
                'class_label': int(label),
                'image_path': img_path
            }
            for i, name in enumerate(BLENDSHAPE_NAMES):
                row[name] = float(blendshapes[i])
                
            records.append(row)
            processed += 1
            if processed % 500 == 0 or processed == total_count:
                print(f"Extracted {processed}/{total_count} images ({processed/total_count*100:.1f}%)")

    df = pd.DataFrame(records)
    
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved blendshapes CSV to: {output_csv}")

    if output_pkl:
        os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
        df.to_pickle(output_pkl)
        print(f"Saved blendshapes Pickle to: {output_pkl}")

    return df