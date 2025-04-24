#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UniGradICON Registration Visualization Script

This script demonstrates how to load medical image data, run registration using the 
UniGradICON model, and visualize the results.

Usage:
  python registration_visualization.py --fixed <fixed_image> --moving <moving_image> [--modality <modality>]

Arguments:
  --fixed        Path to the fixed/target image file (NRRD format)
  --moving       Path to the moving/source image file (NRRD format)
  --modality     Modality of the images (default: mri, options: mri, ct)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import itk
from unigradicon import get_unigradicon

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UniGradICON Registration Visualization")
    parser.add_argument("--fixed", required=True, help="Path to fixed/target image (NRRD)")
    parser.add_argument("--moving", required=True, help="Path to moving/source image (NRRD)")
    parser.add_argument("--modality", default="ct", choices=["mri", "ct"], help="Image modality")
    parser.add_argument("--slice_idx", type=int, default=87, help="Slice index to visualize")
    parser.add_argument("--output", default="registration_result.png", help="Output image path")
    return parser.parse_args()

def load_image(image_path):
    """Load a medical image using ITK."""
    print(f"Loading image: {image_path}")
    image_type = itk.Image[itk.F, 3]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(image_path)
    reader.Update()
    image_data = reader.GetOutput()
    return image_data

def preprocess(img, type="ct"):
    """
    Preprocess medical images for the registration model
    
    Parameters:
    - img: Input image tensor
    - type: Image modality (ct or mri)
    
    Returns:
    - Preprocessed and resized image tensor
    """
    if type == "ct":
        clamp = [-1000, 1000]
        img = (torch.clamp(img, clamp[0], clamp[1]) - clamp[0])/(clamp[1]-clamp[0])
        return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)
    elif type == "mri":
        im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
        img = torch.clip(img, im_min, im_max)
        img = (img-im_min) / (im_max-im_min)
        return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)
    else:
        print(f"Error: Do not support the type {type}")
        return img

def show_as_grid_contour(ax, phi, linewidth=1, stride=8, flip=False):
    """
    Display deformation grid as contour lines
    
    Parameters:
    - ax: Matplotlib axis
    - phi: Deformation field tensor
    - linewidth: Width of contour lines
    - stride: Spacing between contour lines
    - flip: Whether to flip the y-axis
    """
    data_size = phi.size()[1:]
    plot_phi = phi.cpu() - 0.5
    N = plot_phi.size()[-1]
    ax.contour(plot_phi[1], np.linspace(0, N, int(N/stride)), linewidths=linewidth, alpha=0.8)
    ax.contour(plot_phi[0], np.linspace(0, N, int(N/stride)), linewidths=linewidth, alpha=0.8)
    if flip:
        ax.set_ylim([0, data_size[0]])

def show_pair(source, target, warped, phi, axes, idx, flip=False):
    """
    Create a visualization of registration results
    
    Parameters:
    - source: Source image tensor
    - target: Target image tensor
    - warped: Warped source image tensor
    - phi: Deformation field tensor
    - axes: List of matplotlib axes
    - idx: Slice index to visualize
    - flip: Whether to flip the image orientation
    """
    phi_scaled = phi * (torch.tensor((175, 175, 175), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-1)
    origin = 'lower' if flip else 'upper'
    
    # Display images
    axes[0].imshow(source.cpu()[0,0,idx], cmap="gray", origin=origin)
    axes[1].imshow(target.cpu()[0,0,idx], cmap="gray", origin=origin)
    axes[2].imshow(warped.cpu()[0,0,idx], cmap="gray", origin=origin)
    
    # Display target with deformation grid overlay
    axes[3].imshow(target.cpu()[0,0,idx], cmap="gray", origin=origin)
    show_as_grid_contour(axes[3], phi_scaled[0, [1,2], idx], linewidth=0.6, stride=4, flip=flip)
    
    # Display difference images
    axes[4].imshow(target.cpu()[0,0,idx]-source.cpu()[0,0,idx], origin=origin)
    axes[5].imshow(target.cpu()[0,0,idx]-warped.cpu()[0,0,idx], origin=origin)

def visualize_registration_results(source, target, model, slice_idx=87, output_path=None):
    """
    Create and display a complete visualization of registration results
    
    Parameters:
    - source: Source image tensor
    - target: Target image tensor
    - model: Registration model with results
    - slice_idx: Index of the slice to visualize
    - output_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 6, figsize=(15,3))
    show_pair(source, target, model.warped_image_A, model.phi_AB_vectorfield.cpu(), axes, slice_idx)

    # Set titles and remove ticks
    font_size = 12
    titles = ['Source', 'Target', 'Warped', 'Target+Grids', 'Difference Before', 'Difference After']
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=font_size)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    return fig

def main():
    """Main function."""
    args = parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load images
    fixed_image = load_image(args.fixed)
    moving_image = load_image(args.moving)
    
    # Convert ITK images to numpy arrays
    fixed_array = itk.array_from_image(fixed_image)
    moving_array = itk.array_from_image(moving_image)
    
    # Convert numpy arrays to PyTorch tensors and preprocess
    fixed_tensor = preprocess(
        torch.Tensor(fixed_array).unsqueeze(0).unsqueeze(0), 
        type=args.modality
    )
    moving_tensor = preprocess(
        torch.Tensor(moving_array).unsqueeze(0).unsqueeze(0), 
        type=args.modality
    )
    
    # Load UniGradICON model
    print("Loading UniGradICON model...")
    net = get_unigradicon()
    net.to(device)
    net.eval()
    
    # Run registration
    print("Running registration...")
    with torch.no_grad():
        net(moving_tensor.to(device), fixed_tensor.to(device))
    
    # Visualize results
    print("Visualizing results...")
    visualize_registration_results(
        moving_tensor, 
        fixed_tensor, 
        net, 
        slice_idx=args.slice_idx,
        output_path=args.output
    )
    
    print("Done!")

if __name__ == "__main__":
    main()