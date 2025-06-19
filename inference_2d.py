#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Medical Image Registration Visualization Script

Usage:
  python registration_visualization_2d.py --fixed path/to/fixed.png --moving path/to/moving.png [--modality ct|mri]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from unigradicon import get_unigradicon  # must be adapted to accept 2D images

def parse_args():
    parser = argparse.ArgumentParser(description="2D Registration Visualization with UniGradICON")
    parser.add_argument("--fixed", required=True, help="Path to fixed 2D image (e.g., PNG)")
    parser.add_argument("--moving", required=True, help="Path to moving 2D image (e.g., PNG)")
    parser.add_argument("--modality", default="ct", choices=["mri", "ct"], help="Image modality")
    parser.add_argument("--output", default="registration_result.png", help="Output image path")
    return parser.parse_args()

def load_image(image_path):
    """Load a 2D grayscale medical image as a PyTorch tensor."""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("L")
    image_np = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

def preprocess(img, type="ct"):
    if type == "ct":
        clamp = [-1000, 1000]
        img = (torch.clamp(img, clamp[0], clamp[1]) - clamp[0]) / (clamp[1] - clamp[0])
    elif type == "mri":
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img - im_min) / (im_max - im_min)
    else:
        raise ValueError(f"Unsupported modality: {type}")
    return F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)

def show_as_grid_contour(ax, phi, stride=8, flip=False):
    """
    phi: [2, H, W] deformation field
    """
    H, W = phi.shape[1:]
    Y, X = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij')
    U = phi[1, ::stride, ::stride].numpy()
    V = phi[0, ::stride, ::stride].numpy()
    ax.quiver(X, Y, V, U, angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
    if flip:
        ax.set_ylim([H, 0])

def show_pair(source, target, warped, phi, axes, flip=False):
    phi_scaled = phi * (torch.tensor((256, 256), dtype=torch.float32).view(1, 2, 1, 1) - 1)
    origin = 'lower' if flip else 'upper'

    axes[0].imshow(source.cpu()[0, 0], cmap="gray", origin=origin)
    axes[1].imshow(target.cpu()[0, 0], cmap="gray", origin=origin)
    axes[2].imshow(warped.cpu()[0, 0], cmap="gray", origin=origin)

    axes[3].imshow(target.cpu()[0, 0], cmap="gray", origin=origin)
    show_as_grid_contour(axes[3], phi_scaled[0], stride=8, flip=flip)

    axes[4].imshow(target.cpu()[0, 0] - source.cpu()[0, 0], cmap="bwr", origin=origin)
    axes[5].imshow(target.cpu()[0, 0] - warped.cpu()[0, 0], cmap="bwr", origin=origin)

def visualize_registration_results(source, target, model, output_path=None):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    show_pair(source, target, model.warped_image_A, model.phi_AB_vectorfield.cpu(), axes)

    titles = ['Source', 'Target', 'Warped', 'Target+Grid', 'Diff Before', 'Diff After']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    plt.show()
    return fig

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fixed_tensor = preprocess(load_image(args.fixed), type=args.modality)
    moving_tensor = preprocess(load_image(args.moving), type=args.modality)

    # Load model
    print("Loading UniGradICON model...")
    net = get_unigradicon()
    net.to(device)
    net.eval()

    # Run registration
    print("Running registration...")
    with torch.no_grad():
        net(moving_tensor.to(device), fixed_tensor.to(device))

    # Visualize
    print("Visualizing results...")
    visualize_registration_results(moving_tensor, fixed_tensor, net, output_path=args.output)
    print("Done!")

if __name__ == "__main__":
    main()
