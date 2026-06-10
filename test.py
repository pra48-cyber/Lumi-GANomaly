import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.lumi_ganomaly import TriAttentionGenerator
from utils.helpers import LumiAnomalyDataset, tensor_to_pil, generate_heatmap, detect_temporal_window


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation for Lumi-GANomaly Framework")
    parser.add_argument("--test_frames_path", type=str, default="/content/drive/MyDrive/Anomaly/TestData/Test008")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/Anomaly/lumi_generator_checkpoint.pth")
    parser.add_argument("--plot_save_dir", type=str, default="test_results_plots")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--anomaly_threshold", type=float, default=0.49)
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    return parser.parse_args()


def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]


def compute_metrics_with_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.05, seed=42):
    y_true, y_scores = np.array(y_true), np.array(y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    base_auc = auc(fpr, tpr)
    base_eer, _ = calculate_eer(y_true, y_scores)
    
    bootstrapped_aucs, bootstrapped_eers = [], []
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(len(y_true))
    pos_idx, neg_idx = indices[y_true == 1], indices[y_true == 0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"auc": {"estimate": base_auc, "ci_lower": 0.0, "ci_upper": 0.0}, "eer": {"estimate": base_eer, "ci_lower": 0.0, "ci_upper": 0.0}}

    for _ in range(n_bootstraps):
        boot_idx = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True), rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        try:
            boot_fpr, boot_tpr, _ = roc_curve(y_true[boot_idx], y_scores[boot_idx], pos_label=1)
            b_eer, _ = calculate_eer(y_true[boot_idx], y_scores[boot_idx])
            bootstrapped_aucs.append(auc(boot_fpr, boot_tpr))
            bootstrapped_eers.append(b_eer)
        except Exception:
            continue

    low, high = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "auc": {"estimate": base_auc, "ci_lower": np.percentile(bootstrapped_aucs, low), "ci_upper": np.percentile(bootstrapped_aucs, high)},
        "eer": {"estimate": base_eer, "ci_lower": np.percentile(bootstrapped_eers, low), "ci_upper": np.percentile(bootstrapped_eers, high)}
    }


def plot_anomaly_curve(regularity_scores, folder_name, save_path, anomaly_threshold):
    plt.figure(figsize=(12, 6))
    plt.plot(regularity_scores, label="Regularity Score", color="#0077EE", linewidth=1.5)
    plt.axhline(y=anomaly_threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({anomaly_threshold})")
    plt.fill_between(np.arange(len(regularity_scores)), 0, 1, where=regularity_scores < anomaly_threshold, color="red", alpha=0.2, label="Flagged Region")
    plt.xlabel("Frame Number")
    plt.ylabel("Regularity Score")
    plt.title(f"Anomaly Evaluation Curve: {folder_name}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, len(regularity_scores))
    plt.ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_name = os.path.basename(args.test_frames_path)
    save_dir_anomalous = f"test_results_anomalous_frames_{folder_name}"

    model = TriAttentionGenerator().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LumiAnomalyDataset(args.test_frames_path, test_transform, temporal_window=1, enable_enhancement=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    frame_mses, results_for_saving = [], []
    os.makedirs(save_dir_anomalous, exist_ok=True)

    with torch.no_grad():
        for idx, (input_batch, path_batch) in enumerate(tqdm(dataloader, desc="Inference Progress")):
            input_batch = input_batch.to(device)
            recon_batch = model(input_batch)
            inputs_denorm = (input_batch * 0.5) + 0.5
            pred_denorm = (recon_batch * 0.5) + 0.5

            for i in range(input_batch.size(0)):
                error_map = torch.mean((inputs_denorm[i] - pred_denorm[i]) ** 2, dim=0).cpu().numpy()
                mse = np.mean(error_map)
                frame_mses.append(mse)
                results_for_saving.append({"path": path_batch[i], "error_map": error_map, "recon_tensor": recon_batch[i].cpu()})

    frame_mses = np.array(frame_mses)
    psnrs = 10.0 * np.log10(1.0 / (frame_mses + 1e-9))
    min_p, max_p = np.min(psnrs), np.max(psnrs)
    regularity_scores = (psnrs - min_p) / (max_p - min_p + 1e-9) if max_p > min_p else np.ones_like(psnrs)

    plot_save_path = os.path.join(args.plot_save_dir, f"anomaly_curve_{folder_name}.png")
    plot_anomaly_curve(regularity_scores, folder_name, plot_save_path, args.anomaly_threshold)

    mock_gt = (regularity_scores < args.anomaly_threshold).astype(int)
    metrics = compute_metrics_with_ci(mock_gt, 1.0 - regularity_scores, n_bootstraps=args.n_bootstraps)

    print("\n" + "="*60 + f"\n📊 METRIC ANALYSIS WITH CONFIDENCE INTERVALS ({args.n_bootstraps} BOOTSTRAPS)\n" + "="*60)
    print(f"ROC-AUC Estimate : {metrics['auc']['estimate']:.4f} | 95% CI: [{metrics['auc']['ci_lower']:.4f}, {metrics['auc']['ci_upper']:.4f}]")
    print(f"EER Estimate     : {metrics['eer']['estimate']:.4f} | 95% CI: [{metrics['eer']['ci_lower']:.4f}, {metrics['eer']['ci_upper']:.4f}]")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
