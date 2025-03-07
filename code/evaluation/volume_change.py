import nibabel as nib
import numpy as np
import os
import json



def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()  # 获取体素数据
    spacing = img.header.get_zooms()  # 获取spacing信息
    return data, spacing

# 计算体积
def calculate_volume(data, spacing):
    # 找出label=1的体素s
    label_voxels = (data == 1)
    # 计算体素数量
    voxel_count = np.sum(label_voxels)
    # 计算每个体素的体积
    voxel_volume = np.prod(spacing)
    # 计算总的体积
    volume = voxel_count * voxel_volume
    return volume

volume_change = {}
side = ['L', 'R']
dir_path = r'code\data'  # Folder containing your mesh files
IDs = [f for f in os.listdir(dir_path) if f.isdigit()]
output_json_file = r'code\data\evaluation\volume_changes.json'  # Output JSON file
os.makedirs(os.path.dirname(output_json_file), exist_ok=True)


# Process each pair of T1 and T2 meshes
for ID in IDs:
    for s in side:
        input_dir = os.path.join(dir_path, ID,'crop_nii')
        # Construct the full paths for the files
        t1_file_path = os.path.join(input_dir, f'T1_{s}_condyle.nii.gz')
        t2_file_path = os.path.join(input_dir, f'T2_{s}_condyle.nii.gz')

        # Compute the volumes of T1 and T2 meshes
        t1_data, t1_spacing = load_nifti(t1_file_path)
        t2_data, t2_spacing = load_nifti(t2_file_path)

        volume_t1 = calculate_volume(t1_data, t1_spacing)
        volume_t2 = calculate_volume(t2_data, t2_spacing)

        # Calculate the volume change percentage
        volume_change_percentage = ((volume_t2 - volume_t1) / volume_t1) * 100

        # Store the volume change for the current ID, rounded to 2 decimal places
        volume_change[ID] = round(volume_change_percentage, 2)


# Save the results to a JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(volume_change, json_file, indent=4)

