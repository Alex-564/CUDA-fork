import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PathLabelDataset(Dataset):
    """
    Loads images from disk (paths) and produces (tensor, label, clean_path).
    Deterministic: no shuffling; order is the CSV order.
    """

    def __init__(self, samples: List[Tuple[str, int]], input_size: int, interpolation: str):
        # Resizes: forces images to a fixed size
        # TODO: INVESTIGATE CENTERCROP FOR IMAGES TO RETAIN FACIAL DATA
        # Interpolation mode: required to specify how resizing computes new pixels
        #   (BILINEAR is a good default for images)
        # Transforms to tensor: scales to [0,1], changes layouyt to [C,H,W]

        self.samples = samples

        interp_map = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }
        if interpolation not in interp_map:
            raise ValueError(f"Unknown interpolation={interpolation}. Choose from {list(interp_map.keys())}")


        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=interp_map[interpolation]),
            transforms.ToTensor(),  # [0,1] shape: [3,H,W]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Loads image, coverts to RGB, applies transforms
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, int(label), path


def build_cnns(num_cls: int, blur_parameter: float, center_parameter: float, kernel_size: int, seed: int, device: str, same_filter: bool):
    """
    Mirrors CUDA's filter construction from the provided script:
      Conv2d(3,3,kernel_size,groups=3,padding=1).cuda()
      weights ~ Uniform(0, blur_parameter)
      optional single position set to center_parameter
      w[1]=w[0], w[2]=w[0]
    """
    np.random.seed(seed)

    #  Generates per-class conv filters (mirrored from 'final_filter_unlearnable' script)
    cnns = []
    with torch.no_grad():
        for i in range(num_cls):
            conv = torch.nn.Conv2d(3, 3, kernel_size, groups=3, padding=kernel_size // 2).to(device)
            w = np.random.uniform(low=0, high=blur_parameter, size=(3, 1, kernel_size, kernel_size)).astype(np.float32)

            if center_parameter is not None:
                shape = w[0, 0].shape
                w[0, 0, np.random.randint(shape[0]), np.random.randint(shape[1])] = center_parameter

            w[1] = w[0]
            w[2] = w[0]

            conv.weight.copy_(torch.tensor(w, device=device))
            conv.bias.copy_(conv.bias * 0)  # keep bias at 0 
            cnns.append(conv)

    # Debug: All classes use the same filter
    if same_filter and len(cnns) > 0:
        cnns = [cnns[0]] * len(cnns)

    return cnns


def save_tensor_as_image(x: torch.Tensor, out_path: str, image_format: str):
    """
    x: [3,H,W] in [0,1]
    """
    # Clamp to [0,1], move to CPU, permute to [H,W,C], scale to [0,255], convert to uint8
    # save with PIL
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if image_format == "png":
        Image.fromarray(arr).save(out_path, format="PNG")
    elif image_format == "jpg":
        Image.fromarray(arr).save(out_path, format="JPEG", quality=100)
    else:
        raise ValueError(f"Unsupported image_format={image_format}")


def main():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--samples-csv", type=str, required=True)
    p.add_argument("--out-images-dir", type=str, required=True)
    p.add_argument("--out-poison-map", type=str, required=True)
    p.add_argument("--out-metrics-json", type=str, default=None)

    # Generation parameters
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blur-parameter", type=float, default=0.3)
    p.add_argument("--center-parameter", type=float, default=1.0)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--interpolation", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "bicubic", "lanczos"])

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--same-filter", action="store_true")
    p.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])

    args = p.parse_args()

    # Make file paths
    os.makedirs(args.out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_poison_map), exist_ok=True)

    if args.out_metrics_json is None:
        args.out_metrics_json = os.path.join(os.path.dirname(args.out_poison_map), "metrics.json")
    os.makedirs(os.path.dirname(args.out_metrics_json), exist_ok=True)

    # Load samples
    samples: List[Tuple[str, int]] = []
    with open(args.samples_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            samples.append((row["clean_path"], int(row["label"])))

    # Determine num_cls from max label (IMPORTANT NOTE: expects labels are 0..C-1)
    labels = [y for _, y in samples]
    num_cls = max(labels) + 1 if labels else 0
    if num_cls <= 0:
        raise RuntimeError("No samples provided.")

    # Build filters (mirrors CUDA script)
    t0 = time.perf_counter()
    cnns = build_cnns(
        num_cls=num_cls,
        blur_parameter=args.blur_parameter,
        center_parameter=args.center_parameter,
        kernel_size=args.kernel_size,
        seed=args.seed,
        device=args.device,
        same_filter=args.same_filter,
    )

    # Synchronize cuda for accurate time output
    if "cuda" in args.device:
        torch.cuda.synchronize()
    t_filters = time.perf_counter() - t0

    # Dataset + loaders
    ds = PathLabelDataset(samples, input_size=args.input_size, interpolation=args.interpolation)
    dl = DataLoader(ds, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers, 
                    pin_memory=True, 
                    drop_last=False)
    # shuffle=False to keep order deterministic
    # pin_memory=True to speed up host->device transfer
    # drop_last=False to process all samples

    # Timing events
    total_images = 0
    t_compute_gpu = 0.0
    t_save_cpu = 0.0

    # CUDA events for accurate GPU timing (optional if on CPU)
    use_cuda_timing = ("cuda" in args.device) and torch.cuda.is_available()
    if use_cuda_timing:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)


    # Poison generation Loop (PULLED FROM CUDA SCRIPT)
    poison_rows = []
    t_total_start = time.perf_counter()
    torch.set_grad_enabled(False)

    for xb, yb, paths in tqdm(dl, total=len(dl), desc="CUDA generate"):
        # Moves batches to GPU
        xb = xb.to(args.device, non_blocking=True)
        yb = yb.to(args.device, non_blocking=True)

        if use_cuda_timing:
            start_evt.record()

        # Process each sample in batch because each uses its label-specific conv
        out_batch = []
        for i in range(xb.size(0)):
            y = int(yb[i].item())
            x = xb[i:i+1]  # [1,3,H,W]

            out = cnns[y](x).detach()

            # Optional grayscale conversion (same as CUDA code)
            if args.grayscale:
                img_bw = out[0].mean(0)
                out[0, 0] = img_bw
                out[0, 1] = img_bw
                out[0, 2] = img_bw

            # Normalize by max like their script: img/img.max() 
            m = out.max().clamp(min=1e-8)
            out = (out / m).clamp(0, 1)

            out_batch.append(out[0])
        

        if use_cuda_timing:
            end_evt.record()
            torch.cuda.synchronize()
            t_compute_gpu += start_evt.elapsed_time(end_evt) / 1000.0  # ms -> s

        # Save outputs after batch (CPU time)
        t_save_start = time.perf_counter()
        for i, out_img in enumerate(out_batch):
            clean_path = paths[i]
            ext = ".png" if args.image_format == "png" else ".jpg"
            fname = os.path.splitext(os.path.basename(clean_path))[0] + ext
            poisoned_path = os.path.join(args.out_images_dir, fname)

            save_tensor_as_image(out_img, poisoned_path, image_format=args.image_format)
            poison_rows.append((clean_path, poisoned_path))

        t_save_cpu += (time.perf_counter() - t_save_start)
        total_images += len(out_batch)

    t_total = time.perf_counter() - t_total_start

    # Build poison map manifest
    with open(args.out_poison_map, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean_path", "poisoned_path"])
        w.writerows(poison_rows)

    # Metrics
    metrics = {
        "total_images": total_images,
        "total_time_sec": t_total,
        "filters_init_time_sec": t_filters,
        "gpu_compute_time_sec": t_compute_gpu if use_cuda_timing else None,
        "cpu_save_time_sec": t_save_cpu,
        "throughput_img_per_sec": (total_images / t_total) if t_total > 0 else None,
        "args": vars(args),
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved poisoned images: {args.out_images_dir}")
    print(f"Saved poison map: {args.out_poison_map}")
    print(f"Saved metrics: {args.out_metrics_json}")
    print(f"Throughput: {metrics['throughput_img_per_sec']:.2f} img/s")


if __name__ == "__main__":
    main()
