import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.anomaly_detector import Generator
from utils.helpers import AnomalyTestDataset, tensor_to_pil, generate_heatmap, detect_temporal_window


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Evaluation for CBAM Anomaly Detector")
    parser.add_argument("--test_frames_path", type=str, default="/content/drive/MyDrive/Anomaly/TestData/Test008")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/Anomaly/best_anomaly_detector_standard.pth")
    parser.add_argument("--plot_save_dir", type=str, default="test_results_plots_standard")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--mem_dim", type=int, default=2000)
    parser.add_argument("--shrink_thres", type=float, default=0.0025)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--anomaly_threshold", type=float, default=0.49)
    return parser.parse_args()


def plot_anomaly_curve(regularity_scores, folder_name, save_path, anomaly_threshold):
    if len(regularity_scores) == 0:
        print(f"[WARNING] No scores available to plot for {folder_name}.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(regularity_scores, label="Regularity Score", color="#0077EE", linewidth=1.5)

    plt.fill_between(
        np.arange(len(regularity_scores)), 0, 1,
        where=regularity_scores < anomaly_threshold,
        color="red", alpha=0.4, label="Detected Anomaly Region"
    )

    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("Regularity Score (0=Anomalous, 1=Normal)", fontsize=12)
    plt.title(f"Anomaly Detection Analysis: {folder_name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, len(regularity_scores))
    plt.ylim(0, 1.05)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_name = os.path.basename(args.test_frames_path)
    save_dir_anomalous = f"test_results_anomalous_frames_standard_{folder_name}"

    if not os.path.exists(args.model_path) or not os.path.isdir(args.test_frames_path):
        raise FileNotFoundError("Verify that both --model_path and --test_frames_path point to valid locations.")

    model = Generator(args.base_channels, args.mem_dim, args.shrink_thres).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = AnomalyTestDataset(args.test_frames_path, test_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    detect_temporal_window(dataset)

    all_frame_scores = []
    results_for_saving = []
    os.makedirs(save_dir_anomalous, exist_ok=True)

    frame_idx_counter = 0
    with torch.no_grad():
        for input_batch, path_batch in tqdm(dataloader, desc=f"Evaluating {folder_name}"):
            input_batch = input_batch.to(device)
            try:
                recon_batch, _ = model(input_batch)

                for i in range(input_batch.size(0)):
                    current_idx = frame_idx_counter + i
                    input_tensor = input_batch[i]
                    recon_tensor = recon_batch[i]
                    frame_path = path_batch[i]
                    base_name = os.path.basename(frame_path).split(".")[0]

                    input_denorm = (input_tensor * 0.5) + 0.5
                    recon_denorm = (recon_tensor * 0.5) + 0.5
                    pixel_error_map = torch.mean(torch.abs(recon_denorm - input_denorm), dim=0).cpu().numpy()
                    
                    frame_score = np.mean(pixel_error_map)
                    all_frame_scores.append(frame_score)

                    results_for_saving.append({
                        "frame_idx": current_idx,
                        "frame_path": frame_path,
                        "recon_tensor": recon_tensor.cpu(),
                        "error_map": pixel_error_map,
                        "frame_basename": base_name
                    })
            except Exception as e:
                print(f"\n[ERROR] Inference failed processing batch segment: {e}")
                all_frame_scores.extend([np.inf] * input_batch.size(0))

            frame_idx_counter += input_batch.size(0)

    all_frame_scores = np.array(all_frame_scores)
    min_s, max_s = np.min(all_frame_scores), np.max(all_frame_scores)
    
    if max_s > min_s:
        regularity_scores = 1.0 - ((all_frame_scores - min_s) / (max_s - min_s + 1e-9))
    else:
        regularity_scores = np.ones_like(all_frame_scores)

    plot_save_path = os.path.join(args.plot_save_dir, f"anomaly_curve_standard_{folder_name}.png")
    plot_anomaly_curve(regularity_scores, folder_name, plot_save_path, args.anomaly_threshold)

    anomalous_indices = np.where(regularity_scores < args.anomaly_threshold)[0]

    if len(anomalous_indices) > 0:
        saved_count = 0
        for idx in tqdm(anomalous_indices, desc="Saving Anomalous Frames"):
            frame_data = next((item for item in results_for_saving if item["frame_idx"] == idx), None)
            if frame_data:
                base_filename = frame_data["frame_basename"]
                try:
                    orig = Image.open(frame_data["frame_path"]).convert("RGB")
                    orig.resize((args.img_size, args.img_size)).save(os.path.join(save_dir_anomalous, f"{base_filename}_original.png"))

                    tensor_to_pil(frame_data["recon_tensor"]).save(os.path.join(save_dir_anomalous, f"{base_filename}_reconstructed.png"))

                    heatmap_rgb = generate_heatmap(frame_data["error_map"])
                    Image.fromarray(heatmap_rgb).save(os.path.join(save_dir_anomalous, f"{base_filename}_heatmap.png"))
                    saved_count += 1
                except Exception as e:
                    print(f"\n[WARNING] Error saving sequence files for index {idx}: {e}")
        print(f"[SUCCESS] Exported structural components for {saved_count} anomalous frames.")
    else:
        print("[INFO] No frames dropped below the configured anomaly threshold mapping.")


if __name__ == "__main__":
    main()
