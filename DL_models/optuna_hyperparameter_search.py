# -*- coding: utf-8 -*-
"""
Optuna Bayesian Hyperparameter Optimization (HPO) for Upgraded Multimodal Student Engagement Model.
Configuration:
- 20 Trials (n_trials=20)
- Up to 100 Epochs per trial with Early Stopping (patience=5)
- Automated Median Pruning for unpromising trials
- Objective: Maximizes validation balanced accuracy & harmonic score
- Saves best checkpoint and serialized study to DL_models/hyper-1/
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import optuna

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)
sys.path.append(SCRIPT_DIR)

from multimodal_dataloader import get_upgraded_dataloaders
from dual_branch_network import get_upgraded_model
from train_deep_learning import train_epoch, validate_epoch, evaluate_test

HYPER_DIR = os.path.join(SCRIPT_DIR, "hyper-1")
os.makedirs(HYPER_DIR, exist_ok=True)

def objective(trial, args, device):
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("learning_rate", 5e-5, 8e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.45)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    print(f"\n=======================================================", flush=True)
    print(f"  [Optuna Trial {trial.number:02d}/{args.n_trials:02d}] Start", flush=True)
    print(f"  Params -> LR: {lr:.2e}, WD: {weight_decay:.2e}, Dropout: {dropout_rate:.2f}, Batch: {batch_size}", flush=True)
    print(f"=======================================================", flush=True)

    # 2. Dataloaders for this trial
    train_loader, val_loader, test_loader, _ = get_upgraded_dataloaders(
        batch_size=batch_size,
        val_ratio=0.10,
        test_ratio=0.10,
        random_state=args.seed,
        img_size=args.img_size
    )

    # 3. Model construction
    model = get_upgraded_model(
        backbone_type=args.backbone,
        num_classes=3,
        dropout_rate=dropout_rate
    ).to(device)

    # Label smoothing for anti-overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    best_trial_score = 0.0
    patience_counter = 0
    trial_checkpoint_path = os.path.join(HYPER_DIR, f"temp_trial_{trial.number}.pth")

    print(f"\n  =========================================================================================")
    print(f"  >>> TRIAL #{trial.number+1:02d} / {args.n_trials} [LR: {lr:.2e}, Dropout: {dropout:.2f}, Batch: {batch_size}]")
    print(f"  {'Epoch':^10} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10} | {'Val Bal':^10} | {'Time':^7} | {'Status / Checkpoint':<26}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, tr_bal = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_t1 = time.time()

        val_score = 0.5 * val_acc + 0.5 * val_bal

        if val_score > best_trial_score:
            best_trial_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), trial_checkpoint_path)
            status_str = f"* BEST (Val: {val_acc*100:.1f}%)"
        else:
            patience_counter += 1
            status_str = f"Patience: {patience_counter}/{args.patience}"

        ep_str = f"[{epoch:02d}/{args.max_epochs:02d}]"
        time_str = f"{ep_t1 - ep_t0:.1f}s"
        print(f"  {ep_str:^10} | {tr_loss:^10.4f} | {val_loss:^10.4f} | {f'{tr_acc*100:.1f}%':^10} | {f'{val_acc*100:.1f}%':^10} | {f'{val_bal*100:.1f}%':^10} | {time_str:^7} | {status_str:<26}", flush=True)

        # Report to Optuna for pruning
        trial.report(val_score, epoch)
        if trial.should_prune():
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")
            print(f"  -> Trial #{trial.number+1:02d} PRUNED by Optuna MedianPruner at epoch {epoch}.", flush=True)
            if os.path.exists(trial_checkpoint_path):
                os.remove(trial_checkpoint_path)
            raise optuna.exceptions.TrialPruned()

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")
            print(f"  -> Trial #{trial.number+1:02d} Early Stopping at epoch {epoch} (Best score: {best_trial_score:.4f}).", flush=True)
            break
    else:
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}-+-{'-'*26}")

    # Store best score in trial user attributes
    trial.set_user_attr("best_val_score", best_trial_score)
    trial.set_user_attr("checkpoint_path", trial_checkpoint_path)

    return best_trial_score


def run_optuna_study(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION (HPO)")
    print(f"  Total Trials: {args.n_trials}")
    print(f"  Max Epochs per Trial: {args.max_epochs} (Patience: {args.patience})")
    print(f"  Backbone: {args.backbone.upper()} on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=======================================================\n", flush=True)

    study = optuna.create_study(
        study_name="student_engagement_multimodal_hpo",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )

    study.optimize(lambda t: objective(t, args, device), n_trials=args.n_trials)

    print(f"\n" + "="*65, flush=True)
    print(f"  OPTUNA STUDY COMPLETED! (20 Trials)", flush=True)
    print(f"  Best Trial: #{study.best_trial.number}", flush=True)
    print(f"  Best Validation Score: {study.best_value:.4f}", flush=True)
    print(f"  Best Parameters:", flush=True)
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}", flush=True)
    print(f"="*65 + "\n", flush=True)

    # Save Study Object
    study_path = os.path.join(HYPER_DIR, "optuna_study_upgraded.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Saved Optuna study object to: {study_path}", flush=True)

    # Move best trial weights to best_model_optuna_upgraded.pth
    best_temp_path = study.best_trial.user_attrs.get("checkpoint_path", "")
    final_best_path = os.path.join(HYPER_DIR, "best_model_optuna_upgraded.pth")

    if os.path.exists(best_temp_path):
        os.replace(best_temp_path, final_best_path)
        print(f"Saved optimal model weights to: {final_best_path}", flush=True)

    # Clean up any leftover temporary trial weights
    for f in os.listdir(HYPER_DIR):
        if f.startswith("temp_trial_") and f.endswith(".pth"):
            try:
                os.remove(os.path.join(HYPER_DIR, f))
            except Exception:
                pass

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for Student Engagement")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs per trial (default: 100)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "resnet"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_optuna_study(args)
