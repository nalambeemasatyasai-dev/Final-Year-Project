# -*- coding: utf-8 -*-
"""
Main execution script for extracting 52 MediaPipe Face Blendshapes
for all Student Engagement dataset images.
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SCRIPT_DIR)

import mediapipe_facs_extractor as fmp

def main():
    wacv_data_dir = os.path.join(BASE_DIR, "WACV data")
    output_csv = os.path.join(wacv_data_dir, "mediapipe_blendshapes.csv")
    output_pkl = os.path.join(wacv_data_dir, "mediapipe_blendshapes.pkl")
    
    t0 = time.time()
    df = fmp.extract_dataset_blendshapes(
        data_dir=wacv_data_dir,
        output_csv=output_csv,
        output_pkl=output_pkl
    )
    t1 = time.time()
    print(f"\nExtraction completed in {t1 - t0:.2f} seconds!")
    print(f"Total images processed: {len(df)}")
    print(f"Shape: {df.shape}")
    print(f"Features per image: 52 blendshapes + metadata")
    print(df.head(2))

if __name__ == "__main__":
    main()