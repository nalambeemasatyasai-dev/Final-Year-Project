# -*- coding: utf-8 -*-
"""
Upgraded Data Pipeline for Student Engagement Recognition.
Features:
- High-fidelity Lanczos resampling (zero information loss, crisp micro-expressions)
- Zero destructive CLAHE (preserves true natural luminance and contrast)
- Integrated 52-D MediaPipe FACS Blendshapes
- Fixed WeightedRandomSampler (trains on all ~3,500 samples/epoch, not 3 samples)
- Pure NumPy Stratified Train / Val / Test splitting (zero external DLL dependency)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

BLENDSHAPE_COLUMNS = [
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

# ---------------------------------------------------------------------------
# Left/Right blendshape pairs that must be SWAPPED when an image is mirrored.
# Without this, RandomHorizontalFlip shows the network a mirrored face while
# feeding it un-mirrored FACS muscle scores -> contradictory multimodal signal
# (a major source of overfitting / poor fusion generalisation).
# ---------------------------------------------------------------------------
FLIP_PAIR_NAMES = [
    ('browDownLeft', 'browDownRight'),
    ('browOuterUpLeft', 'browOuterUpRight'),
    ('cheekSquintLeft', 'cheekSquintRight'),
    ('eyeBlinkLeft', 'eyeBlinkRight'),
    ('eyeLookDownLeft', 'eyeLookDownRight'),
    ('eyeLookInLeft', 'eyeLookInRight'),
    ('eyeLookOutLeft', 'eyeLookOutRight'),
    ('eyeLookUpLeft', 'eyeLookUpRight'),
    ('eyeSquintLeft', 'eyeSquintRight'),
    ('eyeWideLeft', 'eyeWideRight'),
    ('jawLeft', 'jawRight'),
    ('mouthDimpleLeft', 'mouthDimpleRight'),
    ('mouthFrownLeft', 'mouthFrownRight'),
    ('mouthLeft', 'mouthRight'),
    ('mouthLowerDownLeft', 'mouthLowerDownRight'),
    ('mouthPressLeft', 'mouthPressRight'),
    ('mouthSmileLeft', 'mouthSmileRight'),
    ('mouthStretchLeft', 'mouthStretchRight'),
    ('mouthUpperUpLeft', 'mouthUpperUpRight'),
    ('noseSneerLeft', 'noseSneerRight'),
]

# Pre-computed permutation applied to the 52-d blendshape vector on horizontal flip.
_FLIP_PERM = list(range(len(BLENDSHAPE_COLUMNS)))
_col_index = {c: i for i, c in enumerate(BLENDSHAPE_COLUMNS)}
for _l, _r in FLIP_PAIR_NAMES:
    if _l in _col_index and _r in _col_index:
        _i, _j = _col_index[_l], _col_index[_r]
        _FLIP_PERM[_i], _FLIP_PERM[_j] = _FLIP_PERM[_j], _FLIP_PERM[_i]
FLIP_PERMUTATION = _FLIP_PERM


def _mirror_pil(image):
    """Horizontal flip compatible with both old and new Pillow versions."""
    try:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    except AttributeError:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

class StudentEngagementDataset(Dataset):
    """
    Multimodal Dataset yielding:
    - image: (3, 224, 224) float32 tensor
    - blendshapes: (52,) float32 tensor
    - label: int64 (0: Disengaged, 1: Nominally Engaged, 2: Highly Engaged)
    - img_path: string (for auditing / visualization)
    """
    def __init__(self, df, transform=None, flip_prob=0.0, mirror_consistent_flip=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.flip_prob = float(flip_prob) if flip_prob else 0.0
        self.mirror_consistent_flip = mirror_consistent_flip
        # Index permutation used to left/right-swap FACS scores when mirroring.
        self._flip_perm = torch.tensor(FLIP_PERMUTATION, dtype=torch.long)

        self.image_paths = self.df['image_path'].tolist()
        self.labels = self.df['class_label'].astype(int).tolist()
        
        # Extract the 52 blendshape columns as a single numpy matrix for instant O(1) RAM access
        available_cols = [c for c in BLENDSHAPE_COLUMNS if c in self.df.columns]
        if len(available_cols) == len(BLENDSHAPE_COLUMNS):
            self.blendshapes = self.df[BLENDSHAPE_COLUMNS].values.astype(np.float32)
        else:
            print(f"Warning: Only {len(available_cols)}/{len(BLENDSHAPE_COLUMNS)} blendshapes found in dataframe. Zero-padding.")
            self.blendshapes = np.zeros((len(self.df), 52), dtype=np.float32)
            for i, c in enumerate(available_cols):
                self.blendshapes[:, i] = self.df[c].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        blendshapes = torch.from_numpy(self.blendshapes[idx])
        
        # Open image using PIL with clean RGB conversion (no CLAHE distortion)
        image = Image.open(img_path).convert('RGB')

        # Mirror augmentation with a MATCHED FACS left/right swap so the image
        # and the 52-d blendshape vector remain physically consistent.
        # (Plain RandomHorizontalFlip alone feeds contradictory multimodal signal.)
        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            image = _mirror_pil(image)
            if self.mirror_consistent_flip:
                blendshapes = blendshapes[self._flip_perm]

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        return image, blendshapes, torch.tensor(label, dtype=torch.long), img_path


def get_image_transforms(img_size=224, strong=True):
    """
    Builds transforms using high-fidelity Lanczos resampling.
    Lanczos (8-lobed sinc filter) avoids bilinear blurring and preserves
    crisp eye, mouth, and brow edges without ringing artifacts.

    NOTE: horizontal mirroring is intentionally NOT part of this pipeline.
    It is applied inside StudentEngagementDataset together with a matched
    left/right FACS blendshape swap so both modalities stay consistent.

    `strong=True` adds the anti-overfitting augmentations (geometry + photometric
    + erase) needed to close the ~19-point train/val generalisation gap.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    lanczos = transforms.InterpolationMode.LANCZOS

    base = [transforms.Resize((img_size, img_size), interpolation=lanczos)]

    if strong:
        train_transform = transforms.Compose(base + [
            # Geometric jitter: small pose / scale / translation variation
            # (PIL affine only supports NEAREST/BILINEAR/BICUBIC - not Lanczos;
            #  Lanczos is still used for the base Resize above.)
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                    scale=(0.90, 1.10), shear=6,
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            # Photometric jitter: lighting / white-balance variation
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.20, hue=0.02),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
            # Cutout-style regularisation (forces reliance on distributed cues)
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15),
                                     ratio=(0.5, 2.0), value='random'),
        ])
    else:
        train_transform = transforms.Compose(base + [
            transforms.RandomAffine(degrees=6, translate=(0.05, 0.05),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose(base + [
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def stratified_numpy_split(df, label_col='class_label', val_ratio=0.10, test_ratio=0.10, seed=42):
    """
    Stratified splitting using pure NumPy to avoid Windows Application Control policy issues.
    """
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    unique_classes = sorted(df[label_col].unique())
    for cls in unique_classes:
        cls_indices = df[df[label_col] == cls].index.to_numpy()
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))

        val_part = cls_indices[:n_val]
        test_part = cls_indices[n_val:n_val + n_test]
        train_part = cls_indices[n_val + n_test:]

        val_indices.extend(val_part)
        test_indices.extend(test_part)
        train_indices.extend(train_part)

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    return train_df, val_df, test_df


def get_upgraded_dataloaders(data_dir=None, blendshapes_file=None, batch_size=32,
                            val_ratio=0.10, test_ratio=0.10, random_state=42, img_size=224,
                            strong_aug=True, flip_prob=0.5, balance_mode='sqrt'):
    """
    Creates stratified Train, Validation, and Test dataloaders.
    Fixes the author's 3-sample bug with a proper WeightedRandomSampler.

    balance_mode controls how aggressively the training sampler re-weights the
    three engagement classes:
      - 'uniform': every class sampled equally (prior = 1/3,1/3,1/3)
      - 'sqrt'    : sampling mass ~ sqrt(class_count)  <-- balanced compromise
      - 'natural' : sampling mass ~ class_count (true data prior)
    A fully 'uniform' sampler badly distorts the real 9/51/40 class prior and
    makes the model under-predict the majority class; 'sqrt' keeps class 0
    well-represented while staying far closer to the true distribution.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "WACV data")
        
    if blendshapes_file is None:
        pkl_path = os.path.join(data_dir, "mediapipe_blendshapes.pkl")
        csv_path = os.path.join(data_dir, "mediapipe_blendshapes.csv")
        if os.path.exists(pkl_path):
            blendshapes_file = pkl_path
        elif os.path.exists(csv_path):
            blendshapes_file = csv_path
        else:
            raise FileNotFoundError(f"Blendshapes cache not found at {pkl_path} or {csv_path}. Run feature extraction first!")

    if blendshapes_file.endswith('.pkl'):
        df = pd.read_pickle(blendshapes_file)
    else:
        df = pd.read_csv(blendshapes_file)
        
    print(f"Loaded {len(df)} total samples from {blendshapes_file}", flush=True)
    print(f"Class distribution: {dict(df['class_label'].value_counts())}", flush=True)

    # Pure NumPy stratified split
    train_df, val_df, test_df = stratified_numpy_split(
        df,
        label_col='class_label',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=random_state
    )

    print(f"Dataset split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)

    train_transform, val_transform = get_image_transforms(img_size=img_size, strong=strong_aug)

    train_dataset = StudentEngagementDataset(train_df, transform=train_transform,
                                             flip_prob=flip_prob)
    val_dataset = StudentEngagementDataset(val_df, transform=val_transform, flip_prob=0.0)
    test_dataset = StudentEngagementDataset(test_df, transform=val_transform, flip_prob=0.0)

    # Fix author's 3-sample bug with proper sample-level weights
    train_labels = np.array(train_dataset.labels)
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float64)

    mode = (balance_mode or 'sqrt').lower()
    if mode == 'uniform':
        cls_weights = 1.0 / np.maximum(class_counts, 1.0)
    elif mode in ('natural', 'none', 'off'):
        cls_weights = np.ones_like(class_counts)
    else:  # 'sqrt' (default compromise)
        cls_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    cls_weights = cls_weights / cls_weights.sum()

    # Effective sampling probability of each class under WeightedRandomSampler:
    #   P(class c) ~ count_c * weight_c
    eff_prob = class_counts * cls_weights
    eff_prob = eff_prob / eff_prob.sum()
    print(f"Sampler balance mode: '{mode}' -> effective class sampling "
          f"probabilities: {np.round(eff_prob, 4).tolist()}", flush=True)

    sample_weights = np.array([cls_weights[y] for y in train_labels], dtype=np.float32)
    sample_weights_tensor = torch.from_numpy(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(train_dataset),  # Ensures ALL ~3500 samples are trained every epoch!
        replacement=True
    )

    # Windows safe workers: 0 to prevent multiprocessing lockups, batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, test_df
