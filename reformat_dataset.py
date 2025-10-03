import os
import shutil

# Dynamically get home directory
home_dir = os.path.expanduser("~")

# Define correct paths
source_dir = os.path.join(home_dir, "Federated-fMRI/fmri_data/ds004192-download")
formatted_dir = os.path.join(home_dir, "Federated-fMRI/fmri_data/formatted_data")

# Create formatted output directory
os.makedirs(formatted_dir, exist_ok=True)

# Loop through and process each subject
for i, folder_name in enumerate(sorted(os.listdir(source_dir))):
    folder_path = os.path.join(source_dir, folder_name)
    if not os.path.isdir(folder_path) or not folder_name.startswith("sub-"):
        continue

    sub_id = f"sub-{i+1:02d}"
    new_sub_path = os.path.join(formatted_dir, sub_id)
    os.makedirs(new_sub_path, exist_ok=True)

    found = False
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                src_nii_path = os.path.join(root, file)
                dst_nii_path = os.path.join(new_sub_path, f"{sub_id}_bold.nii.gz")
                shutil.copy(src_nii_path, dst_nii_path)
                print(f"✔ Copied {src_nii_path} → {dst_nii_path}")
                found = True
                break
        if found:
            break

    if not found:
        print(f"⚠️ No NIfTI files found for: {folder_name}")

print("\n✅ All subjects formatted in:", formatted_dir)
