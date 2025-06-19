import numpy as np
import nibabel as nib
import imageio
import os

def save_2d_image_as_3d_nii(image_path, output_path):
    # Read the image (grayscale assumed)
    image_2d = imageio.imread(image_path, as_gray=True)
    
    # Convert to float32 (recommended for NIfTI)
    image_2d = image_2d.astype(np.float32)
    
    # Unsqueeze to add a singleton dimension: shape becomes (H, W, 1)
    image_3d = np.expand_dims(image_2d, axis=-1)

    # Create a NIfTI image
    affine = np.eye(4)  # identity affine, adjust if needed
    nii_img = nib.Nifti1Image(image_3d, affine)

    # Save as .nii.gz
    nib.save(nii_img, output_path)
    print(f"Saved 3D NIfTI image to: {output_path}")

# Example usage
if __name__ == "__main__":
    input_image = "2.jpg"
    output_nii = "2.nii.gz"
    save_2d_image_as_3d_nii(input_image, output_nii)
