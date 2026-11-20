import argparse
import copy
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.g3.models import build_g3_model, unfreeze_deeper_layers


def get_case_insensitive_path(base_path: str, relative_path: str):
    parts = relative_path.replace("\\", "/").strip("/").split("/")
    current_path = base_path
    for part in parts:
        if not os.path.exists(current_path):
            return None
        try:
            items = os.listdir(current_path)
        except OSError:
            return None
        matched = [item for item in items if item.lower() == part.lower()]
        if matched:
            current_path = os.path.join(current_path, matched[0])
        else:
            return None
    return current_path


class GemstoneSubset(Dataset):
    def __init__(self, df, image_dir, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.resolved_paths = []
        valid_rows = []

        for idx in range(len(self.df)):
            rel_path = self.df.loc[idx, "path"]
            if rel_path.startswith("Gem Stone Images Dataset/"):
                rel_path = rel_path.replace("Gem Stone Images Dataset/", "", 1)
            actual = get_case_insensitive_path(image_dir, rel_path)
            if actual and os.path.exists(actual):
                valid_rows.append(idx)
                self.resolved_paths.append(actual)

        self.df = self.df.iloc[valid_rows].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.resolved_paths[idx]).convert("RGB")
        label = self.class_to_idx[self.df.loc[idx, "class"]]
        if self.transform:
            img = self.transform(img)
        return img, label, self.df.loc[idx, "path"]


def get_transforms():
    # Color-safe augmentation: NO hue or saturation jitter
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def make_optimizer(model_name, stage, head_params, backbone_params=None, weight_decay=1e-4):
    if model_name in ["densenet169", "convnext_small"]:
        if stage == "A":
            return optim.SGD(head_params, lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        return optim.SGD(
            [
                {"params": backbone_params, "lr": 1e-3},
                {"params": head_params, "lr": 1e-3},
            ],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if model_name == "mobilenetv2":
        if stage == "A":
            return optim.RMSprop(head_params, lr=1e-3, weight_decay=weight_decay)
        return optim.RMSprop(
            [
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-4},
            ],
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
    return total_loss / total, (correct / total) * 100.0


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.classes_txt, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    full_df = pd.read_csv(args.csv_path)
    train_all_df = full_df[full_df["split"] == "train"].reset_index(drop=True)
    test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)

    if "fold" in train_all_df.columns and train_all_df["fold"].notna().all():
        val_df = train_all_df[train_all_df["fold"] == 0].reset_index(drop=True)
        trn_df = train_all_df[train_all_df["fold"] != 0].reset_index(drop=True)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        trn_idx, val_idx = next(skf.split(train_all_df, train_all_df["class"]))
        trn_df = train_all_df.iloc[trn_idx].reset_index(drop=True)
        val_df = train_all_df.iloc[val_idx].reset_index(drop=True)

    train_tf, eval_tf = get_transforms()
    train_ds = GemstoneSubset(trn_df, args.img_dir, class_to_idx, transform=train_tf)
    val_ds = GemstoneSubset(val_df, args.img_dir, class_to_idx, transform=eval_tf)
    test_ds = GemstoneSubset(test_df, args.img_dir, class_to_idx, transform=eval_tf)

    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    for seed in args.seeds:
        t0 = time.time()
        print(f"\n================ {args.model} | Seed {seed} ================")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        model, head_params = build_g3_model(args.model, num_classes=len(class_names), pretrained=True)
        model = model.to(device)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())

        # Stage A: Frozen backbone (5 epochs)
        opt_a = make_optimizer(args.model, "A", head_params)
        for epoch in range(1, args.stage_a_epochs + 1):
            model.train()
            trn_loss = 0.0
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                opt_a.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                opt_a.step()
                trn_loss += loss.item() * inputs.size(0)
            val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
            print(f"[Stage A | Ep {epoch:02d}/{args.stage_a_epochs}] Train Loss: {trn_loss/len(train_ds):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())

        # Stage B: Unfreeze deeper layers (up to 25 epochs + early stopping)
        backbone_params, head_params = unfreeze_deeper_layers(model, args.model)
        opt_b = make_optimizer(args.model, "B", head_params, backbone_params)
        no_improve = 0
        for epoch in range(1, args.stage_b_epochs + 1):
            model.train()
            trn_loss = 0.0
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                opt_b.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                opt_b.step()
                trn_loss += loss.item() * inputs.size(0)
            val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
            print(f"[Stage B | Ep {epoch:02d}/{args.stage_b_epochs}] Train Loss: {trn_loss/len(train_ds):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at Stage B epoch {epoch}")
                    break

        model.load_state_dict(best_weights)
        torch.save(model.state_dict(), os.path.join(args.models_dir, f"g3_{args.model}_rgb_s{seed}.pt"))

        model.eval()
        predictions_list = []
        with torch.no_grad():
            for inputs, labels, paths in test_loader:
                inputs = inputs.to(device)
                probs = torch.softmax(model(inputs), dim=1).cpu().numpy()[0]
                row = {"path": paths[0], "true": class_names[labels.item()], "where": "test"}
                for i, cls_name in enumerate(class_names):
                    row[cls_name] = probs[i]
                predictions_list.append(row)

        df_preds = pd.DataFrame(predictions_list)
        csv_out = os.path.join(args.preds_dir, f"g3_{args.model}_rgb_s{seed}.csv")
        df_preds.to_csv(csv_out, index=False)

        probs_mat = df_preds[class_names].values
        true_idx = df_preds["true"].map(class_to_idx).values
        top1 = (probs_mat.argmax(axis=1) == true_idx).mean() * 100
        top5 = np.mean([t in np.argsort(p)[-5:] for t, p in zip(true_idx, probs_mat)]) * 100
        print(f"Saved: {csv_out} ({(time.time() - t0)/60:.1f} min) | Test Top-1: {top1:.2f}% | Test Top-5: {top5:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["densenet169", "convnext_small", "mobilenetv2"])
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--csv-path", default="splits/chow68.csv")
    parser.add_argument("--classes-txt", default="splits/classes.txt")
    parser.add_argument("--preds-dir", default="results/preds")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage-a-epochs", type=int, default=5)
    parser.add_argument("--stage-b-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    run_training(parser.parse_args())