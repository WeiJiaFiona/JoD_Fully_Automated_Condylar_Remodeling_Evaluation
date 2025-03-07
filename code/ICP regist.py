import torchio as tio  
import os 
import time  
from glob import glob  
import numpy as np 
import torch 
from skimage.measure import marching_cubes  
import open3d as o3d  
import trimesh  
from tqdm import tqdm  
from pymeshlab import MeshSet, Mesh  
import argparse  
import shutil

# Function to save 3D mesh data to a file
def save_mesh(verts, faces, path, invert_faces=False):
    mesh = Mesh(verts, faces)  
    ms = MeshSet()  
    ms.add_mesh(mesh)  
    if invert_faces:  
        ms.invert_faces_orientation()  
    ms.save_current_mesh(path)  

# Function to register target ROI
def warped_target_with_roi(mesh_dir,save_dir,ID, side,  sample_points_registration=100000):
    
    T1_ramus_reference_path =  os.path.join(mesh_dir,f'T1_{side}_ramus.off')
    T2_ramus_reference_path =  os.path.join(mesh_dir,f'T2_{side}_ramus.off')
    T1_condyle_mesh_path =  os.path.join(mesh_dir,f'T1_{side}_condyle.off')
    T2_condyle_mesh_path =  os.path.join(mesh_dir,f'T2_{side}_condyle.off')


    # If the reference or target path files do not exist, print a message and return
    if not os.path.exists(T1_ramus_reference_path ) or not os.path.exists(T2_ramus_reference_path ) or not os.path.exists( T1_condyle_mesh_path ) or not os.path.exists(T2_condyle_mesh_path):
        print('the target is missing')
        return 
    
    # Load reference and target mesh files
    T1_ramus_img = trimesh.load_mesh(T1_ramus_reference_path)  
    T2_ramus_img = trimesh.load_mesh(T2_ramus_reference_path)
    T1_condyle_img = trimesh.load_mesh(T1_condyle_mesh_path)
    T2_condyle_img = trimesh.load_mesh(T2_condyle_mesh_path)
    
    # Sample points from the reference mesh for registration
    points1 =   T1_ramus_img.sample(sample_points_registration)
    points2 = T2_ramus_img.sample(sample_points_registration)
    
    # Compute the centroids of the reference and target point clouds and perform translation
    centroid_1 = np.mean(points1, axis=0)  # Calculate the centroid of the reference point cloud
    centroid_2 = np.mean(points2, axis=0)  # Calculate the centroid of the target point cloud
    

    centroid_shift = centroid_1 - centroid_2  # Calculate the centroid shift

    points2 += centroid_shift  # Translate the reference point cloud to the target point cloud location
    
    # Perform ICP registration using the point clouds
    source_pcd = o3d.geometry.PointCloud()  # Create a source point cloud object
    source_pcd.points = o3d.utility.Vector3dVector(points2)  # Convert reference point cloud to Open3D point cloud
    source_pcd.estimate_normals()  # Estimate normals of the point cloud
    
    target_pcd = o3d.geometry.PointCloud()  # Create a target point cloud object
    target_pcd.points = o3d.utility.Vector3dVector(points1)  # Convert target point cloud to Open3D point cloud
    target_pcd.estimate_normals()  # Estimate normals of the point cloud

    # ICP process, using point-to-plane transformation estimation
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=50,  # Maximum correspondence distance
        init=np.eye(4),  # Initial transformation matrix
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # Use point-to-plane transformation estimation
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)  # Maximum iteration limit for ICP
    )
    
    # Apply the registration transformation to the reference and target meshes
    moving_mesh = o3d.geometry.TriangleMesh()  # Create a target mesh object


    moving_mesh.vertices = o3d.utility.Vector3dVector(T2_condyle_img.vertices + centroid_shift)  # Translate the target mesh
    moving_mesh.triangles = o3d.utility.Vector3iVector(T2_condyle_img.faces)  # Get the target mesh's face data
    
    
    # Apply the ICP transformation matrix to the reference and target meshes
    moving_mesh.transform(reg_p2l.transformation)
    
    # Save the transformed meshes
    save_mesh(moving_mesh.vertices, moving_mesh.triangles, os.path.join(save_dir,  f'T2_regist_{side}.off'))
    shutil.copy2(T1_condyle_mesh_path,  os.path.join(save_dir,  f'T1_reference_{side}.off'))


tps = ['T1', 'T2']
surgery_sides = 'R'

dir_path = r'code\data'
IDs = os.listdir(dir_path)


for ID in IDs:
    for tp in tps:
        mesh_dir = os.path.join(dir_path,ID,'crop_mesh')
        save_dir = os.path.join(dir_path,ID,'registrated_output')
        os.makedirs(save_dir,exist_ok=True)

        warped_target_with_roi(mesh_dir,save_dir,ID, side=surgery_sides, sample_points_registration=100000)
