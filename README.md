# reconall-to-wavefront

A Python pipeline to convert FreeSurfer `recon-all` output into Wavefront OBJ files for neuroimaging visualization.

![alt text](https://github.com/AlexanderAtalay/reconall-to-blender/blob/banner.jpg "White surface rendered in Blender")

## Overview

This tool processes FreeSurfer's cortical surface meshes (e.g., pial, white matter) and subcortical segmentation volumes (aseg) from a completed `recon-all` run, converting them into Wavefront OBJ format meshes for 3D visualization and rendering.

## Features

- Converts cortical surfaces to OBJ meshes
- Extracts subcortical regions from aseg segmentation as 3D meshes
- Supports customizable region selection
- Applies transformations (rotation, scaling, translation) for optimal visualization
- Pure Python implementation using nibabel, numpy, scipy, and scikit-image

## Usage

See the script's help: `python reconall_to_wavefront.py --help`

## Requirements

- nibabel
- numpy
- scipy
- scikit-image