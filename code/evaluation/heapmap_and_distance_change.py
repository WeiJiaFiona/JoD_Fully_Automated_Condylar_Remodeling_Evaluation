import trimesh
import numpy as np
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import json


# Function to process paired meshes and generate heatmap
def generate_heatmap_and_cal_distance(mesh1_path, mesh2_path, output_folder, file_id,spacing):
    # Read the OFF files for T1 and T2 meshes
    mesh1 = trimesh.load(mesh1_path)  # 术前髁突
    mesh2 = trimesh.load(mesh2_path)  # 术后髁突

    # Create KDTree for T2 vertices
    tree = KDTree(mesh2.vertices)

    # Query the nearest points from T1 vertices to T2 vertices
    distances, indices = tree.query(mesh1.vertices)

    # Get the nearest points from T2
    nearest_points = mesh2.vertices[indices]

    # Compute signed distance (negative for inward, positive for outward)
    normals = mesh1.vertex_normals
    vectors = nearest_points - mesh1.vertices
    signed_distances = np.sum(vectors * normals, axis=1)

    signed_distances_mm = signed_distances * np.linalg.norm(spacing)

    # Set color mapping using the actual physical distance range
    vmin = np.min(signed_distances_mm) 
    vmax = np.max(signed_distances_mm) 
    cmap = plt.get_cmap('coolwarm') 
    norm = Normalize(vmin=vmin, vmax=vmax)  

    # Map the distances to colors
    colors = cmap(norm(signed_distances_mm))

    # Visualize the T1 mesh with the color applied
    mesh1.visual.vertex_colors = (colors[:, :3] * 255).astype(np.uint8)

    # Save the colorized mesh as a PLY file
    ply_output_path = os.path.join(output_folder, f"{file_id}_heapmap.ply")
    mesh1.export(ply_output_path)
    distance_change = [vmin, vmax]

    return distance_change

# Set the input directory and output directory
dir_path = r'code\data' 
output_base_folder = r'code\data\evaluation\heapmap'
os.makedirs(output_base_folder, exist_ok=True)

# List all the mesh files in the directory
IDs = [f for f in os.listdir(dir_path) if f.isdigit()]
side = ['L', 'R']

distance_change = {}
# Process each pair of T1 and T2 meshes and generate the heatmaps
for ID in IDs:
    for s in side:
        
        t1_file = os.path.join(dir_path, ID,'registrated_output',f'T1_reference_{s}.off')
        t2_file = os.path.join(dir_path, ID,'registrated_output',f'T2_regist_{s}.off')

        if not os.path.exists(t1_file) or not os.path.exists(t2_file):
            continue
        
        
        # Generate heatmap and save results
        distance = generate_heatmap_and_cal_distance(t1_file, t2_file, output_base_folder, ID,spacing=[0.3,0.3,0.3])
        distance_change[ID] = distance

output_json_file = r'code\data\evaluation\distance_changes.json'  # Output JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(distance_change, json_file, indent=4)
print("Heatmaps and distance change statistics saved successfully.")