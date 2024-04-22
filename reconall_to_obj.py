#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib


with open('/autofs/space/nicc_003/users/xander/code/neuroviz/aseg_tools/region_labels.json', 'r') as label_json_path:
    region_labels = json.load(label_json_path)

with open('/autofs/space/nicc_003/users/xander/code/neuroviz/aseg_tools/aseg_labels.json', 'r') as aseg_label_path:
    aseg_labels = json.load(aseg_label_path)


def execute_command(command, log=False, silent=True):
    """ 
    Executes a shell command and optionally logs the output to a file.

    Args:
        command (str): The shell command to execute.
        log (str, optional): The file path to log the output. Defaults to False.
        silent (bool, optional): If no log is provided, silence output print. Defaults to False.

    Returns:
        None
    """
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')

        if log:
            with open(log, 'a') as log_path:
                log_path.write(f'{output}\n\n')
        elif not silent:
            print(output,'\n')
            
    except subprocess.CalledProcessError as e:
        if log:
            with open(log, 'a') as log:
                log.write(f'Error running: \n{command}\n\n')
                log.write(f'{e.output}\n')
        else:
            print(output,'\n')


def write_region_surface(surface, vertices, output_path):
    """
    Write the surface of a specific lobe to a file.

    Args:
    surface (tuple): The vertices and faces of the surface.
    vertices (array): Indices of vertices that belong to the region.
    output_path (str): Path to save the new surface file.
    """
    verts, faces = surface
    region_verts = verts[vertices]
    # Create a mapping from old vertex indices to new indices
    old_to_new = {old: new for new, old in enumerate(vertices)}
    # Re-map faces to new indices
    region_faces = np.array([[old_to_new[vert] for vert in face if vert in old_to_new] for face in faces if all(v in old_to_new for v in face)])
    # Write the new surface
    nib.freesurfer.io.write_geometry(output_path, region_verts, region_faces)

def extract_region_surfaces(recon_all_dir, output_dir, region_labels):
    """
    Extracts and writes out pial surface files for each lobe based on the DKT cortical atlas.
    
    Args:
    pial_path (str): Path to the hemisphere pial surface file.
    annot_path (str): Path to the hemisphere .annot file from FreeSurfer.
    output_dir (str): Directory to store the resulting lobe surface files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for hemi in ['rh', 'lh']:
        pial_path = f'{recon_all_dir}/surf/{hemi}.pial'
        annot_path = f'{recon_all_dir}/label/{hemi}.aparc.annot'
    
        # Load the annotation and the surface
        labels, ctab, names = nib.freesurfer.io.read_annot(annot_path)
        surface = nib.freesurfer.io.read_geometry(pial_path)
    
        # Map from names to vertices
        name_to_vertices = {name.decode('ascii').lower(): np.where(labels == idx)[0] for idx, name in enumerate(names)}
    
        # Extract surfaces for each lobe
        for parent_region, regions in region_labels.items():
            vertices = np.unique(np.concatenate([name_to_vertices[region] for region in regions if region in name_to_vertices]))
            write_region_surface(surface, vertices, os.path.join(output_dir, f"{hemi}.{parent_region}"))


def rotate_mesh(object_path, degrees):
    """
    Rotate the mesh in an OBJ file around the x-axis and write to a new OBJ file.
    
    Parameters:
    - input_path (str): Path to the original OBJ file.
    """
    # Rotation matrix for 90 degrees around the x-axis
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])
    
    with open(object_path, 'r') as file:
        lines = file.readlines()
    
    with open(object_path, 'w') as file:
        for line in lines:
            if line.startswith('v '):  # Only rotate vertices
                parts = line.strip().split()
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rotated_vertex = vertex.dot(rotation_matrix)
                file.write(f"v {rotated_vertex[0]} {rotated_vertex[1]} {rotated_vertex[2]}\n")
            else:
                file.write(line)  # Write other lines (faces, etc.) unchanged

def scale_mesh(object_path, scale_factor):
    """
    Scale the mesh in an OBJ file by a given factor and write to a new OBJ file.

    Parameters:
    - input_path (str): Path to the original OBJ file.
    - output_path (str): Path to the output OBJ file with scaled vertices.
    - scale_factor (float): Factor by which to scale the mesh vertices.
    """
    with open(object_path, 'r') as file:
        lines = file.readlines()

    with open(object_path, 'w') as file:
        for line in lines:
            if line.startswith('v '):  # Only scale vertices
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                scaled_vertex = [v * scale_factor for v in vertex]
                file.write(f"v {scaled_vertex[0]} {scaled_vertex[1]} {scaled_vertex[2]}\n")
            else:
                file.write(line)  # Write other lines (faces, etc.) unchanged

def translate_mesh(object_path, x, y, z):
    """
    Translate all vertices in a Wavefront OBJ file by specified x, y, and z offsets.

    Parameters:
    input_path (str): Path to the input OBJ file.
    output_path (str): Path to the output OBJ file with translated vertices.
    x (float): The translation offset along the x-axis.
    y (float): The translation offset along the y-axis.
    z (float): The translation offset along the z-axis.
    """
    try:
        with open(object_path, 'r') as file:
            lines = file.readlines()

        with open(object_path, 'w') as file:
            for line in lines:
                if line.startswith('v '):  # Vertex data starts with 'v '
                    parts = line.split()
                    # Translate vertex coordinates
                    new_x = float(parts[1]) + x
                    new_y = float(parts[2]) + y
                    new_z = float(parts[3]) + z
                    # Write the new vertex line
                    file.write(f"v {new_x} {new_y} {new_z}\n")
                else:
                    # Write all other lines as is
                    file.write(line)
    except Exception as e:
        print(f"An error occurred: {e}")

def combine_asc_files(asc_paths, output_asc_path):
    """
    Combine ASCII files from left and right hemispheres, adjust the indices of the faces, and write to a single ASCII file.

    Parameters:
    - ascii_paths (list): List containing paths to the ASCII files for 'lh' and 'rh'.
    - output_asc_path (str): Path to output the combined ASCII file.
    """
    vertices = []
    faces = []
    offset = 0  # Vertex index offset for the right hemisphere

    for ascii_path in asc_paths:
        with open(ascii_path, 'r') as file:
            data = file.readlines()

        # Skip header and extract vertex/face count
        header = data.pop(0)  # Skip the header
        vertex_count, face_count = map(int, data.pop(0).split())

        # Append vertices directly
        vertices.extend(data[:vertex_count])

        # Adjust and append face indices
        adjusted_faces = []
        for face in data[vertex_count:]:
            indices = face.split()[:-1]  # Ignore the last number which is always 0
            adjusted_indices = [str(int(index) + offset) for index in indices] + ['0\n']  # Adjust indices and re-add the trailing 0
            adjusted_faces.append(' '.join(adjusted_indices))

        faces.extend(adjusted_faces)
        offset += vertex_count  # Update the offset for the next hemisphere

    # Write combined data to ASCII file
    with open(output_asc_path, 'w') as asc_file:
        asc_file.write("# Combined ASCII file generated from FreeSurfer surfaces\n")
        asc_file.write(f"{len(vertices)} {len(faces)}\n")  # Write the total counts
        asc_file.writelines(vertices)
        asc_file.writelines(faces)


def extract_label(recon_all_dir, label_index, output_file):
    """Extract a specific label from aseg file and save it as a binary volume."""
    input_file = f"{recon_all_dir}/mri/aseg.mgz"
    
    if type(label_index) == list:
        join_niftis = []
        for label in label_index:
            tmp_file = output_file.replace(output_file.split('/')[-1], label + '.nii.gz')
            command = f"mri_binarize --i {input_file} --match {label} --o {tmp_file}"
            execute_command(command)
            join_niftis.append(tmp_file)

        command = f"fslmaths {join_niftis[0]} -add " + " -add ".join(join_niftis[1:]) + f" {output_file}"
        execute_command(command)

    else:
        command = f"mri_binarize --i {input_file} --match {label_index} --o {output_file}"
        execute_command(command)
            
    command = f"fslmaths {output_file} -mul 1 {output_file}"
    execute_command(command)

def nii2obj(input_nifti, output_object):
    command = f"nii2mesh {input_nifti} -l 0 {output_object}"
    execute_command(command)

def convert_aseg_to_surface(recon_all_dir, output_dir, aseg_label, label_index):
    os.makedirs(f"{output_dir}/tmp", exist_ok=True)
    input_path = f"{recon_all_dir}/mri/aseg.mgz"
    nifti_output = f"{output_dir}/tmp/{aseg_label}.nii.gz"
    output_object = f"{output_dir}/{aseg_label}.obj"

    extract_label(recon_all_dir, label_index, nifti_output)
    nii2obj(nifti_output, output_object)
    rotate_mesh(output_object, 90)
    scale_mesh(output_object, 0.1)
    translate_mesh(output_object, -0.26, -0.5, -3.2)
    
    shutil.rmtree(f'{output_dir}/tmp')


def convert_srf_to_obj(input_surface, output_object):
    """
    Converts a surface file in ASCII format (SRF) to a Wavefront OBJ file format.

    This function reads vertex and face data from an SRF file and writes them
    as vertices (v) and faces (f) in the OBJ file format. The indices in OBJ
    files are 1-based, so adjustments are made accordingly during conversion.

    Parameters:
    input_surface (str): The file path of the input SRF file. It should contain
                         vertex and face information starting from the second line.
    output_object (str): The file path where the converted OBJ file will be written.
                         This file will include vertices and face indices formatted
                         according to the OBJ specification.

    Each line of the ASCII file should follow the structure:
    - Second line: two integers representing the number of vertices (nV) and faces (nF).
    - Subsequent nV lines: three floating-point numbers representing vertex coordinates.
    - Following nF lines: three integers representing the vertex indices of each face.
    """
    # Read lines of srf file
    with open(input_surface, 'r') as srf:
        lines = srf.readlines()

    with open(output_object, 'w') as obj:
        # Read number of vertices and faces from the second line
        nV, nF = map(int, lines[1].strip().split())

        # Process vertices
        for i in range(2, nV+2):
            coords = lines[i].strip().split()
            obj.write(f"v {coords[0]} {coords[1]} {coords[2]}\n")

        # Process faces
        for i in range(nV+2, nV+nF+2):
            indices = lines[i].strip().split()
            obj.write(f"f {int(indices[0])+1} {int(indices[1])+1} {int(indices[2])+1}\n")


def convert_bilateral_srf_to_obj(surface, input_dir, output_dir, log=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tmp'), exist_ok=True)
    combined_srf_path = os.path.join(output_dir, 'tmp', f"{surface}.srf")
    combined_obj_path = os.path.join(output_dir, f"{surface}.obj")
    
    surface_hemis = {}
    for hemi in ['rh', 'lh']:
        surface_hemis[hemi] = os.path.join(output_dir, f"{hemi}.{surface}.asc")
        execute_command(f'mris_convert {input_dir}/surf/{hemi}.{surface} {surface_hemis[hemi]}', log=log, silent=True)

    # Concatenate ASCII files into one .srf file
    combine_asc_files(surface_hemis.values(), combined_srf_path)
        
    # Convert the combined .srf file to .obj
    convert_srf_to_obj(combined_srf_path, combined_obj_path)
    rotate_mesh(combined_obj_path, 90)
    scale_mesh(combined_obj_path, 0.1)
    translate_mesh(combined_obj_path, 0, -2.0, -1.5)
    
    for srf_path in surface_hemis.values():
        os.remove(srf_path)
    os.remove(combined_srf_path)
    shutil.rmtree(f'{output_dir}/tmp')


def reconall_to_objects(recon_all_dir, output_dir, log=False):
    """
    Converts FreeSurfer surface files to a single combined Wavefront OBJ format suitable for use in Blender.
    
    Parameters:
    - recon_all_dir (str): Path to the directory containing FreeSurfer's 'recon-all' output.
    - output_dir (str): Path where the combined converted OBJ files will be saved.

    Both hemispheres for each surface type (pial, white) are combined into single files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the surfaces to process
    surfaces = ['pial', 'white']
    cortical_regions = ['frontal', 'motor', 'prefrontal', 'parietal', 'temporal', 'occipital']
    aseg_regions = ['Cerebellum-Exterior','Cerebellum-White-Matter','Cerebellum-Cortex','Thalamus','Caudate','Putamen','Pallidum',
                    'Brain-Stem','Hippocampus','Amygdala','Insula','Operculum','Lesion','Accumbens-area','VentralDC','Claustrum']

    for surface in surfaces:
        convert_bilateral_srf_to_obj(surface, 
                                     recon_all_dir, 
                                     f'{output_dir}/surf', 
                                     log)

    extract_region_surfaces(recon_all_dir, f'{output_dir}/tmp/surf', region_labels=region_labels)
    for region in cortical_regions:
        convert_bilateral_srf_to_obj(region, 
                                     f'{output_dir}/tmp',
                                     f'{output_dir}/cortical', 
                                     log)

    for region in aseg_regions:
        indices = aseg_labels[region]
        surface_file = region.lower().replace('-','_')
    
        try:
            convert_aseg_to_surface(recon_all_dir,
                                   f'{output_dir}/aseg',
                                   surface_file,
                                   indices)
        except:
            None
        
    shutil.rmtree(f'{output_dir}/tmp')


def main():
    parser = argparse.ArgumentParser(description="Convert FreeSurfer surface files to OBJ format for Blender.")
    parser.add_argument("-i", "--input", type=str, help="Path to the directory containing FreeSurfer's 'recon-all' output.")
    parser.add_argument("-o", "--output", type=str, help="Output directory path where the OBJ files will be saved.")
    parser.add_argument("-l", "--log", type=str, help="Path to the log file for debugging.", default=None)

    args = parser.parse_args()

    reconall_to_objects(args.recon_all_dir, args.output_dir, args.log)

if __name__ == "__main__":
    main()
