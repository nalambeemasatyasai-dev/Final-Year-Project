# -*- coding: utf-8 -*-
"""
Training script for the Upgraded Dual-Branch Student Engagement Model.
Implements:
- Full GPU training on NVIDIA GeForce RTX 2050
- Fixed full-epoch balanced sampler (training on all ~3,500 samples/epoch)
- Label smoothing (0.1) for anti-overfitting
- Cosine Annealing Learning Rate Schedule
- Early Stopping with checkpointing to DL_models/hyper-1/best_model_upgraded.pth
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)
sys.path.append(SCRIPT_DIR)

from multimodal_dataloader import get_upgraded_dataloaders
from dual_branch_network import get_upgraded_model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, blendshapes, labels, _ in train_loader:
        images = images.to(device, non_blocking=True)
        blendshapes = blendshapes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, blendshapes)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, epoch_bal_acc


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, blendshapes, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            blendshapes = blendshapes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images, blendshapes)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return val_loss, val_acc, val_bal_acc, all_preds, all_labels


def evaluate_test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, blendshapes, labels, _ in test_loader:
            images = images.to(device, non_blocking=True)
            blendshapes = blendshapes.to(device, non_blocking=True)
            
            outputs = model(images, blendshapes)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["0: Disengaged", "1: Nominally Engaged", "2: Highly Engaged"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, bal_acc, report, cm, all_probs, all_preds, all_labels


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"  Upgraded Student Engagement Model Training")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Backbone: {args.backbone}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"  Label Smoothing: 0.1 | Regularizer: Dropout 0.25 + LayerNorm")
    print(f"=======================================================\n")

    train_loader, val_loader, test_loader, test_df = get_upgraded_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        img_size=args.img_size
    )

    # Save test set dataframe for standalone verification
    os.makedirs(os.path.join(SCRIPT_DIR, "hyper-1"), exist_ok=True)
    test_df_path = os.path.join(SCRIPT_DIR, "hyper-1", "test_split_unseen.pkl")
    test_df.to_pickle(test_df_path)
    print(f"Saved holdout test split to: {test_df_path}")

    image_only = getattr(args, 'image_only', False)
    model = get_upgraded_model(
        backbone_type=args.backbone,
        num_classes=3,
        dropout_rate=args.dropout,
        image_only=image_only
    ).to(device)

    # Anti-overfitting: Label smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_score = 0.0
    best_epoch = 0
    image_only = getattr(args, 'image_only', False)
    ckpt_name = f"best_model_image_only_{args.backbone}.pth" if image_only else f"best_model_upgraded_{args.backbone}.pth"
    checkpoint_path = os.path.join(SCRIPT_DIR, "hyper-1", ckpt_name)

    history = {
        'train_loss': [], 'train_acc': [], 'train_bal_acc': [],
        'val_loss': [], 'val_acc': [], 'val_bal_acc': []
    }

    start_time = time.time()

    print(f"\n  {'Epoch':^10} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10} | {'Val Bal':^10} | {'Time':^7} | {'Status / Checkpoint':<26}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}", flush=True)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, tr_bal = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_t1 = time.time()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_bal_acc'].append(tr_bal)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(val_bal)

        # Primary optimization metric: Harmonic blend of val accuracy and balanced accuracy
        val_score = 0.5 * val_acc + 0.5 * val_bal

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_bal_acc': val_bal,
                'backbone': args.backbone,
                'img_size': args.img_size
            }, checkpoint_path)
            status_str = f"* BEST (Val: {val_acc*100:.1f}%)"
        else:
            patience_counter += 1
            status_str = f"Patience: {patience_counter}/{args.patience}"

        ep_str = f"[{epoch:02d}/{args.epochs:02d}]"
        time_str = f"{ep_t1 - ep_t0:.1f}s"
        print(f"  {ep_str:^10} | {tr_loss:^10.4f} | {val_loss:^10.4f} | {f'{tr_acc*100:.1f}%':^10} | {f'{val_acc*100:.1f}%':^10} | {f'{val_bal*100:.1f}%':^10} | {time_str:^7} | {status_str:<26}", flush=True)

        if patience_counter >= args.patience:
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")
            print(f"  -> Early Stopping triggered at epoch {epoch} (Best epoch was {best_epoch} with val score {best_val_score:.4f}).", flush=True)
            break
    else:
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")

    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.1f}s (~{total_train_time/60:.2f} mins).", flush=True)

    # Load best checkpoint and evaluate on Unseen Test Set
    print(f"\n--- Loading Best Checkpoint from {checkpoint_path} ---")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_acc, test_bal_acc, report, cm, _, y_true, y_pred = evaluate_test(model, test_loader, device)

    print("\n" + "="*55)
    print("        FINAL TEST SET EVALUATION REPORT (UNSEEN DATA)")
    print("="*55)
    print(f"Overall Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Balanced Test Accuracy:      {test_bal_acc * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("="*55)

    # Dynamic update of Results/DL/evaluation.csv with live test metrics
    dl_results_dir = os.path.join(BASE_DIR, "Results", "DL")
    os.makedirs(dl_results_dir, exist_ok=True)
    eval_csv = os.path.join(dl_results_dir, "dl_models_evaluation.csv")
    image_only = getattr(args, 'image_only', False)
    if image_only:
        model_tag = f"Image_Only_{args.backbone.capitalize()}"
    else:
        model_tag = f"Dual_Branch_{args.backbone.capitalize()}_MediaPipe"
    try:
        p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if os.path.exists(eval_csv):
            eval_df = pd.read_csv(eval_csv, index_col=0)
            eval_df = eval_df[eval_df['Model'] != model_tag]
            new_row = pd.DataFrame([{
                'Model': model_tag,
                'Accuracy': test_acc,
                'Precision': p_macro,
                'Recall': r_macro,
                'F1': f1_macro
            }])
            eval_df = pd.concat([eval_df, new_row], ignore_index=True)
            eval_df.to_csv(eval_csv)
            print(f"Updated {eval_csv} with live test metrics for {model_tag}.")
        else:
            eval_df = pd.DataFrame([{
                'Model': model_tag,
                'Accuracy': test_acc,
                'Precision': p_macro,
                'Recall': r_macro,
                'F1': f1_macro
            }])
            eval_df.to_csv(eval_csv)
            print(f"Created {eval_csv} with live test metrics for {model_tag}.")
    except Exception as e:
        print(f"Notice: Failed to update dl_models_evaluation.csv: {e}")

    # Save history and metrics
    hist_name = f"history_image_only_{args.backbone}.pkl" if image_only else f"history_upgraded_{args.backbone}.pkl"
    history_path = os.path.join(SCRIPT_DIR, "hyper-1", hist_name)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to: {history_path}")

    # Generate and save DL plots to Results/DL/
    dl_results_dir = os.path.join(BASE_DIR, "Results", "DL")
    os.makedirs(dl_results_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_prefix = f"{args.backbone}_image_only" if image_only else args.backbone

        # 1. Training curves (Loss & Accuracy)
        fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(history['train_loss']) + 1)

        ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', lw=1.8, markersize=4)
        ax1.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss', lw=1.8, markersize=4)
        ax1.set_title(f'Loss Curves ({plot_prefix.upper()})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(epochs_range, [a * 100 for a in history['train_acc']], 'b-o', label='Train Acc (%)', lw=1.8, markersize=4)
        ax2.plot(epochs_range, [a * 100 for a in history['val_acc']], 'r-s', label='Val Acc (%)', lw=1.8, markersize=4)
        ax2.plot(epochs_range, [a * 100 for a in history['val_bal_acc']], 'g--^', label='Val Balanced Acc (%)', lw=1.8, markersize=4)
        ax2.set_title(f'Accuracy Curves ({plot_prefix.upper()})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        curves_png = os.path.join(dl_results_dir, f"training_curves_{plot_prefix}.png")
        curves_pdf = os.path.join(dl_results_dir, f"training_curves_{plot_prefix}.pdf")
        fig_curves.savefig(curves_png, dpi=150, bbox_inches='tight')
        fig_curves.savefig(curves_pdf, bbox_inches='tight')
        plt.close(fig_curves)
        print(f"Saved training curves to: {curves_png}")

        # 2. Confusion Matrix Heatmap
        fig_cm, ax = plt.subplots(figsize=(7, 6))
        class_names = ["Disengaged (0)", "Nominally Engaged (1)", "Highly Engaged (2)"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix: {plot_prefix.upper()} on Unseen Test Data', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        plt.tight_layout()

        cm_png = os.path.join(dl_results_dir, f"confusion_matrix_{plot_prefix}.png")
        cm_pdf = os.path.join(dl_results_dir, f"confusion_matrix_{plot_prefix}.pdf")
        fig_cm.savefig(cm_png, dpi=150, bbox_inches='tight')
        fig_cm.savefig(cm_pdf, bbox_inches='tight')
        plt.close(fig_cm)
        print(f"Saved confusion matrix to: {cm_png}")

    except Exception as e:
        print(f"Notice: Plot generation skipped: {e}")

    return {
        'backbone': args.backbone,
        'test_acc': test_acc,
        'test_bal_acc': test_bal_acc,
        'report': report,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Upgraded Student Engagement Model")
    parser.add_argument("--backbone", type=str, default="efficientnet",
                        choices=["efficientnet", "efficientnet_b0", "efficientnet_b1", "resnet", "resnet18", "resnet34", "resnet50", "mobilenet_v3", "mobilenet_v3_large", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_only", action="store_true", help="Train Image-Only architecture (like Riju IF)")
    args = parser.parse_args()

    run_training(args)
