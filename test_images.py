import numpy as np
import itk
import matplotlib.pyplot as plt
target_path = "RegLib_C01_1.nrrd"
source_path = "RegLib_C01_2.nrrd"

target_itk = itk.imread(target_path)
target_meta = dict(target_itk)
target = np.asarray(target_itk)
print(f"Target shape: {target.shape}")
print(f"Target spacing: {target_meta['spacing']}")
print(f"Target direction: {target_meta['direction']}")

source_itk = itk.imread(source_path)
source_meta = dict(source_itk)
source = np.asarray(source_itk)
print(f"Source shape: {source.shape}")
print(f"Source spacing: {source_meta['spacing']}")
print(f"Source direction: {source_meta['direction']}")

# Check whether the orientation of the images are the same.
assert np.array_equal(dict(target_itk)["direction"], dict(source_itk)["direction"]), "The orientation of source and target images need to be the same."

fig, axes = plt.subplots(1,2)
axes[0].imshow(source[100])
axes[0].set_title('Source')
axes[1].imshow(target[100])
axes[1].set_title('Target')
plt.show()
