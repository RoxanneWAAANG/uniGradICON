import SimpleITK as sitk

# Read the .nii.gz
image = sitk.ReadImage("/home/jack/Projects/yixin-llm/yixin-llm-data/uniGradICON/LungCT/LungCT_0023_0001.nii.gz")
# Write out as .nrrd
sitk.WriteImage(image, "/home/jack/Projects/yixin-llm/yixin-llm-data/uniGradICON/LungCT/LungCT_0023_0001.nrrd")
