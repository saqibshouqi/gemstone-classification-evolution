import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.g3.models import build_g3_model
from src.g3.train import get_case_insensitive_path


def get_target_layers(model, model_name: str):
    if model_name == "densenet169":
        return [model.features.norm5]
    if model_name == "convnext_small":
        return [model.features[7][-1]]
    if model_name == "convnext_small_cbam":
        return [model.features[8]]
    if model_name == "mobilenetv2":
        return [model.features[-1]]
    raise ValueError(f"Unknown G3 model_name: {model_name}")


def pick_diverse_top5(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = sub_df.sort_values("conf", ascending=False)
    selected, seen = [], set()
    for _, row in sub_df.iterrows():
        if row["true"] not in seen:
            selected.append(row)
            seen.add(row["true"])
        if len(selected) == 5:
            break
    return pd.DataFrame(selected)


def run_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.classes_txt, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.fig_dir, exist_ok=True)
    df = pd.read_csv(args.pred_csv)
    probs = df[class_names].values
    df["pred_idx"] = probs.argmax(axis=1)
    df["pred"] = [class_names[i] for i in df["pred_idx"]]
    df["conf"] = probs.max(axis=1)
    df["correct"] = df["pred"] == df["true"]

    correct_5 = pick_diverse_top5(df[df["correct"]])
    wrong_5 = pick_diverse_top5(df[~df["correct"]])

    model, _ = build_g3_model(args.model, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

    target_layers = get_target_layers(model, args.model)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    fig.suptitle(
        f"Grad-CAM: {args.model} (Seed {args.seed}) — Top: 5 Correct | Bottom: 5 Wrong",
        fontsize=16,
        fontweight="bold",
    )

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for row_idx, (group_name, subset) in enumerate([("correct", correct_5), ("wrong", wrong_5)]):
            for col_idx, (_, item) in enumerate(subset.reset_index(drop=True).iterrows()):
                rel = item["path"].replace("Gem Stone Images Dataset/", "", 1)
                img_path = get_case_insensitive_path(args.img_dir, rel)
                pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
                rgb_float = np.float32(pil_img) / 255.0
                input_tensor = eval_tf(pil_img).unsqueeze(0).to(device)

                grayscale_cam = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(int(item["pred_idx"]))],
                )[0, :]
                overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

                ax = axes[row_idx, col_idx]
                ax.imshow(overlay)
                ax.axis("off")
                color = "darkgreen" if group_name == "correct" else "crimson"
                ax.set_title(
                    f"True: {item['true']}\nPred: {item['pred']} ({item['conf']*100:.1f}%)",
                    fontsize=10,
                    color=color,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(args.fig_dir, f"gradcam_{args.model}_s{args.seed}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM figure: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["densenet169", "convnext_small", "convnext_small_cbam", "mobilenetv2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--classes-txt", default="splits/classes.txt")
    parser.add_argument("--fig-dir", default="results/figures")
    parser.add_argument("--seed", type=int, default=0)
    run_gradcam(parser.parse_args())