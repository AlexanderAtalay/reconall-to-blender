#!/usr/bin/env python3
"""
reconall_to_wavefront.py

Convert FreeSurfer recon-all output to Wavefront OBJ files.
All processing is done using nibabel, numpy, scipy, and scikit-image.

Usage:
    python reconall_to_wavefront.py -i /path/to/sub-XYZ [options]
    python reconall_to_wavefront.py --list-aseg-regions

Requirements:
    nibabel, numpy, scipy, scikit-image
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from skimage import measure as sk_measure
except ImportError:
    print("Error: scikit-image is required. Install with: pip install scikit-image", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.sparse import csr_matrix
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Resource paths (JSON label files co-located with this script)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ASEG_LABELS_PATH = SCRIPT_DIR / "aseg_tools" / "aseg_labels.json"
REGION_LABELS_PATH = SCRIPT_DIR / "aseg_tools" / "region_labels.json"

# Default set of subcortical regions (curated subset used in the original pipeline)
DEFAULT_ASEG_REGIONS = [
    "Cerebellum-Exterior", "Cerebellum-White-Matter", "Cerebellum-Cortex",
    "Thalamus", "Caudate", "Putamen", "Pallidum", "Brain-Stem",
    "Hippocampus", "Amygdala", "Insula", "Operculum", "Lesion",
    "Accumbens-area", "VentralDC", "Claustrum",
]

DEFAULT_SURFACES = ["pial", "white"]

# ===========================================================================
# OBJ I/O
# ===========================================================================

def write_obj(vertices, faces, path, name=None):
    """
    Write a mesh to a Wavefront OBJ file.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates.
    faces : ndarray, shape (M, 3)
        Triangle indices, zero-indexed.
    path : str
        Output file path.
    name : str, optional
        Object name written as a comment header.
    """
    with open(path, "w") as f:
        if name:
            f.write(f"# {name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ format is 1-indexed
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


# ===========================================================================
# Mesh transforms
# ===========================================================================

def apply_rotation_x(vertices, degrees):
    """Rotate vertices around the x-axis (same convention as original pipeline)."""
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation matrix applied as  v @ R  (row-vector convention, matching original)
    R = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ])
    return vertices @ R


def apply_scale(vertices, factor):
    """Uniformly scale vertices."""
    return vertices * factor


def apply_translation(vertices, x, y, z):
    """Translate vertices by (x, y, z)."""
    return vertices + np.array([x, y, z], dtype=float)


def transform_vertices(vertices, rotation=90.0, scale=0.1, translation=(0.0, 0.0, 0.0)):
    """Apply the standard rotation → scale → translation pipeline."""
    v = apply_rotation_x(vertices, rotation)
    v = apply_scale(v, scale)
    v = apply_translation(v, *translation)
    return v


# ===========================================================================
# Volume preprocessing
# ===========================================================================

def clean_binary_volume(vol_data, closing_radius=1):
    """
    Apply morphological closing to clean up binary segmentation.
    
    Closing = dilation followed by erosion.
    - Fills small holes
    - Removes small bridges and noise
    
    Parameters
    ----------
    vol_data : ndarray, dtype uint8
        Binary volume mask
    closing_radius : int
        Radius of the structuring element (default: 1)
    
    Returns
    -------
    cleaned : ndarray, dtype uint8
    """
    if not _HAS_SCIPY:
        return vol_data
    
    struct = ndi.generate_binary_structure(3, connectivity=2)
    # Binary closing: dilate then erode
    closed = ndi.binary_closing(vol_data.astype(bool), structure=struct)
    return closed.astype(np.uint8)


# ===========================================================================
# Laplacian smoothing
# ===========================================================================

def laplacian_smooth(vertices, faces, iterations=5, lam=0.5):
    """
    Laplacian smoothing via sparse-matrix adjacency averaging.

    Falls back to a Python loop if scipy is unavailable (much slower for large meshes).

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (M, 3)
    iterations : int
    lam : float  — blending weight (0 = no change, 1 = full Laplacian step)

    Returns
    -------
    smoothed vertices : ndarray, shape (N, 3)
    """
    if iterations == 0:
        return vertices

    n = len(vertices)
    verts = vertices.copy()

    if _HAS_SCIPY:
        # Build symmetric adjacency matrix from all directed edges of each triangle
        rows = np.concatenate([
            faces[:, 0], faces[:, 1], faces[:, 2],
            faces[:, 1], faces[:, 2], faces[:, 0],
        ])
        cols = np.concatenate([
            faces[:, 1], faces[:, 2], faces[:, 0],
            faces[:, 0], faces[:, 1], faces[:, 2],
        ])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        degree = np.asarray(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0  # avoid division by zero

        for _ in range(iterations):
            neighbor_avg = adj.dot(verts) / degree[:, np.newaxis]
            verts = verts + lam * (neighbor_avg - verts)
    else:
        # Pure-Python fallback (slow for large meshes)
        adjacency = [set() for _ in range(n)]
        for f in faces:
            adjacency[f[0]].update((f[1], f[2]))
            adjacency[f[1]].update((f[0], f[2]))
            adjacency[f[2]].update((f[0], f[1]))
        for _ in range(iterations):
            new_verts = verts.copy()
            for i, nbrs in enumerate(adjacency):
                if nbrs:
                    new_verts[i] = verts[i] + lam * (np.mean(verts[list(nbrs)], axis=0) - verts[i])
            verts = new_verts

    return verts


# ===========================================================================
# FreeSurfer surface I/O
# ===========================================================================

def read_surface(path):
    """
    Read a FreeSurfer binary surface file.

    Returns vertices in RAS space (CRAS offset applied) and zero-indexed faces.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (M, 3)
    """
    vertices, faces, metadata = nib.freesurfer.read_geometry(str(path), read_metadata=True)
    cras = metadata.get("cras", np.zeros(3))
    return (vertices + cras).astype(np.float64), faces.astype(np.int32)


def combine_hemispheres(lh_verts, lh_faces, rh_verts, rh_faces):
    """
    Merge left- and right-hemisphere meshes into a single mesh.

    Right-hemisphere face indices are offset by the number of left-hemisphere vertices.

    Returns
    -------
    vertices : ndarray, shape (N_lh + N_rh, 3)
    faces    : ndarray, shape (M_lh + M_rh, 3)
    """
    offset = len(lh_verts)
    combined_verts = np.vstack([lh_verts, rh_verts])
    combined_faces = np.vstack([lh_faces, rh_faces + offset])
    return combined_verts, combined_faces


# ===========================================================================
# Cortical region extraction  (replaces mri_extract_region + mris_convert)
# ===========================================================================

def extract_cortical_region(surface_path, annot_path, region_names):
    """
    Extract the subset of a hemisphere's surface belonging to named cortical regions.

    Uses the Desikan–Killiany atlas (.aparc.annot) by default (controlled by caller).

    Parameters
    ----------
    surface_path : str or Path
        E.g. subject/surf/lh.pial
    annot_path : str or Path
        E.g. subject/label/lh.aparc.annot
    region_names : list[str]
        Region names as they appear in the annotation (case-insensitive).

    Returns
    -------
    vertices : ndarray, shape (N, 3)  or  None
    faces    : ndarray, shape (M, 3)  or  None
    """
    labels, _ctab, names = nib.freesurfer.io.read_annot(str(annot_path))
    verts, faces = read_surface(surface_path)

    # Build name → annotation-index lookup (bytes decoded, lowercased)
    name_to_idx = {
        name.decode("ascii").lower(): idx
        for idx, name in enumerate(names)
    }

    vertex_sets = []
    for rname in region_names:
        key = rname.lower()
        if key in name_to_idx:
            vertex_sets.append(np.where(labels == name_to_idx[key])[0])

    if not vertex_sets:
        return None, None

    vertex_set = np.unique(np.concatenate(vertex_sets))

    # Remap: original vertex index → new (compact) index
    old_to_new = np.full(len(verts), -1, dtype=np.int32)
    old_to_new[vertex_set] = np.arange(len(vertex_set), dtype=np.int32)

    # Keep only faces where every vertex is within the region
    valid = np.all(old_to_new[faces] >= 0, axis=1)
    new_faces = old_to_new[faces[valid]]
    new_verts = verts[vertex_set]

    return new_verts, new_faces


# ===========================================================================
# Volume → mesh
# ===========================================================================

def extract_aseg_binary(aseg_data, label_indices):
    """
    Create a binary mask for one or more ASeg label indices.

    Parameters
    ----------
    aseg_data : ndarray
        Integer-valued segmentation volume.
    label_indices : int | str | list[int|str]

    Returns
    -------
    mask : ndarray, dtype uint8  (same shape as aseg_data; 1 = label present)
    """
    if not isinstance(label_indices, list):
        label_indices = [label_indices]
    mask = np.zeros(aseg_data.shape, dtype=np.uint8)
    for idx in label_indices:
        mask |= (aseg_data == int(idx)).astype(np.uint8)
    return mask


def volume_to_mesh(vol_data, affine, smooth_iterations=15):
    """
    Convert a binary 3-D volume to a triangle mesh using Marching Cubes.

      1. Apply morphological closing to clean up the volume (fill holes, remove artifacts).
      2. Pad volume by 1 voxel on each side (ensures closed surface at borders).
      3. Run Marching Cubes at level = 0.5 (between 0 and 1 for a binary mask).
      4. Apply the volume affine to transform voxel coords → RAS space.
      5. Optionally apply Laplacian smoothing.

    Parameters
    ----------
    vol_data : ndarray, shape (X, Y, Z), dtype uint8
    affine   : ndarray, shape (4, 4)   — voxel-to-RAS affine from nibabel
    smooth_iterations : int            — 0 disables smoothing

    Returns
    -------
    vertices : ndarray, shape (N, 3)  or  None  (if volume is empty)
    faces    : ndarray, shape (M, 3)  or  None
    """
    if vol_data.max() == 0:
        return None, None

    # Clean up the binary volume to remove artifacts and fill holes
    vol_data = clean_binary_volume(vol_data)

    # Pad to prevent clipping at volume boundaries
    padded = np.pad(vol_data, pad_width=1, mode="constant", constant_values=0)

    verts_vox, faces, _normals, _values = sk_measure.marching_cubes(
        padded, level=0.5, step_size=1, allow_degenerate=False
    )

    # Undo padding offset (subtract 1 from each voxel coordinate)
    verts_vox -= 1.0

    # Apply affine: voxel → RAS
    ones = np.ones((len(verts_vox), 1))
    verts_hom = np.hstack([verts_vox, ones])          # (N, 4)
    verts_ras = (affine @ verts_hom.T).T[:, :3]       # (N, 3)

    if smooth_iterations > 0:
        verts_ras = laplacian_smooth(verts_ras, faces, iterations=smooth_iterations)

    return verts_ras, faces.astype(np.int32)


# ===========================================================================
# High-level processing functions
# ===========================================================================

def process_surfaces(recon_dir, output_dir, surfaces, split_hemispheres,
                     rotation, scale, translation, verbose):
    """Export whole cortical/white-matter surfaces."""
    surf_out = os.path.join(output_dir, "surf")
    os.makedirs(surf_out, exist_ok=True)

    for surface in surfaces:
        if verbose:
            print(f"  Surface: {surface}")
        lh_path = os.path.join(recon_dir, "surf", f"lh.{surface}")
        rh_path = os.path.join(recon_dir, "surf", f"rh.{surface}")

        if not os.path.isfile(lh_path) or not os.path.isfile(rh_path):
            print(f"  Warning: surface files not found for {surface}, skipping.", file=sys.stderr)
            continue

        lh_v, lh_f = read_surface(lh_path)
        rh_v, rh_f = read_surface(rh_path)

        if split_hemispheres:
            for tag, v, f in [("lh", lh_v, lh_f), ("rh", rh_v, rh_f)]:
                tv = transform_vertices(v, rotation, scale, translation)
                write_obj(tv, f, os.path.join(surf_out, f"{tag}.{surface}.obj"),
                          name=f"{tag}.{surface}")
        else:
            v, f = combine_hemispheres(lh_v, lh_f, rh_v, rh_f)
            tv = transform_vertices(v, rotation, scale, translation)
            write_obj(tv, f, os.path.join(surf_out, f"{surface}.obj"), name=surface)


def process_cortical_regions(recon_dir, output_dir, region_labels_map, cortical_regions,
                              split_hemispheres, rotation, scale, translation, verbose):
    """Export labelled cortical parcellation regions."""
    ctx_out = os.path.join(output_dir, "cortical")
    os.makedirs(ctx_out, exist_ok=True)

    for region in cortical_regions:
        if region not in region_labels_map:
            if verbose:
                print(f"  Skipping unknown cortical region: {region}")
            continue

        if verbose:
            print(f"  Cortical region: {region}")

        region_names = region_labels_map[region]

        lh_v, lh_f = extract_cortical_region(
            os.path.join(recon_dir, "surf", "lh.pial"),
            os.path.join(recon_dir, "label", "lh.aparc.annot"),
            region_names,
        )
        rh_v, rh_f = extract_cortical_region(
            os.path.join(recon_dir, "surf", "rh.pial"),
            os.path.join(recon_dir, "label", "rh.aparc.annot"),
            region_names,
        )

        if split_hemispheres:
            if lh_v is not None:
                tv = transform_vertices(lh_v, rotation, scale, translation)
                write_obj(tv, lh_f, os.path.join(ctx_out, f"lh.{region}.obj"),
                          name=f"lh.{region}")
            if rh_v is not None:
                tv = transform_vertices(rh_v, rotation, scale, translation)
                write_obj(tv, rh_f, os.path.join(ctx_out, f"rh.{region}.obj"),
                          name=f"rh.{region}")
        else:
            if lh_v is not None and rh_v is not None:
                v, f = combine_hemispheres(lh_v, lh_f, rh_v, rh_f)
            elif lh_v is not None:
                v, f = lh_v, lh_f
            elif rh_v is not None:
                v, f = rh_v, rh_f
            else:
                if verbose:
                    print(f"    No vertices found for region: {region}")
                continue
            tv = transform_vertices(v, rotation, scale, translation)
            write_obj(tv, f, os.path.join(ctx_out, f"{region}.obj"), name=region)


def process_aseg_regions(recon_dir, output_dir, aseg_labels_map, aseg_regions,
                         split_hemispheres, smooth_iterations,
                         rotation, scale, translation, verbose):
    """Export subcortical ASeg segmentation regions."""
    aseg_out = os.path.join(output_dir, "aseg")
    os.makedirs(aseg_out, exist_ok=True)

    aseg_path = os.path.join(recon_dir, "mri", "aseg.mgz")
    if not os.path.isfile(aseg_path):
        print(f"Error: aseg.mgz not found at {aseg_path}", file=sys.stderr)
        return

    aseg_img = nib.load(aseg_path)
    aseg_data = np.asarray(aseg_img.dataobj)

    for region in aseg_regions:
        if region not in aseg_labels_map:
            if verbose:
                print(f"  Skipping unknown aseg region: {region}")
            continue

        if verbose:
            print(f"  ASeg region: {region}")

        label_info = aseg_labels_map[region]
        safe_name = region.lower().replace("-", "_").replace("/", "_")
        is_bilateral = isinstance(label_info, list)

        try:
            if is_bilateral and split_hemispheres:
                # Write separate lh / rh files
                for tag, idx in [("lh", label_info[0]), ("rh", label_info[1])]:
                    mask = extract_aseg_binary(aseg_data, [idx])
                    v, f = volume_to_mesh(mask, aseg_img.affine, smooth_iterations)
                    if v is not None:
                        tv = transform_vertices(v, rotation, scale, translation)
                        write_obj(tv, f, os.path.join(aseg_out, f"{tag}.{safe_name}.obj"),
                                  name=f"{tag}.{safe_name}")
            else:
                # Combined bilateral, or midline structure
                indices = label_info if isinstance(label_info, list) else [label_info]
                mask = extract_aseg_binary(aseg_data, indices)
                v, f = volume_to_mesh(mask, aseg_img.affine, smooth_iterations)
                if v is not None:
                    tv = transform_vertices(v, rotation, scale, translation)
                    write_obj(tv, f, os.path.join(aseg_out, f"{safe_name}.obj"),
                              name=safe_name)
                elif verbose:
                    print(f"    No voxels found for {region} — skipping.")
        except Exception as exc:
            print(f"  Warning: failed to process aseg region '{region}': {exc}",
                  file=sys.stderr)


# ===========================================================================
# Utilities
# ===========================================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_aseg_labels_from_lut(lut_path):
    """
    Parse FreeSurferColorLUT.txt to create aseg labels dictionary.
    """
    aseg_region_dictionary = {}
    with open(lut_path) as f:
        lines = f.readlines()[1:]  # Skip header
        aseg_conversions = dict([line.split()[:2] for line in lines if line.split()])
    
    for index, region in aseg_conversions.items():
        if 'Left-' in region or 'Right-' in region:
            region_label = region.replace('Left-', '').replace('Right-','').lower()
            if region_label not in aseg_region_dictionary:
                aseg_region_dictionary[region_label] = [int(index)]
            else:
                aseg_region_dictionary[region_label].append(int(index))
        else:
            aseg_region_dictionary[region.lower()] = int(index)
    
    return aseg_region_dictionary


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="reconall_to_wavefront",
        description="Convert FreeSurfer recon-all output to Wavefront OBJ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Convert all regions (output to <subject>/wavefront/)
  %(prog)s -i /path/to/sub-XYZ

  # Specify output directory
  %(prog)s -i /path/to/sub-XYZ -o /path/to/output

  # Select specific aseg regions
  %(prog)s -i /path/to/sub-XYZ --aseg-regions Hippocampus Amygdala Thalamus

  # Export all aseg regions defined in aseg_labels.json
  %(prog)s -i /path/to/sub-XYZ --all-aseg

  # Split each bilateral structure into lh/rh files
  %(prog)s -i /path/to/sub-XYZ --split-hemispheres

  # Surfaces only, no aseg or cortical regions
  %(prog)s -i /path/to/sub-XYZ --no-cortical --no-aseg

  # List available aseg region names
  %(prog)s --list-aseg-regions
""",
    )

    # -- Input / output -------------------------------------------------------
    io = parser.add_argument_group("input / output")
    io.add_argument(
        "-i", "--input", metavar="SUBJECT_DIR",
        help="Path to FreeSurfer recon-all subject directory (required unless using --list-*).",
    )
    io.add_argument(
        "-o", "--output", metavar="OUTPUT_DIR", default=None,
        help="Directory to write OBJ files. Default: <subject_dir>/wavefront/",
    )

    # -- What to process ------------------------------------------------------
    sel = parser.add_argument_group("region selection")
    sel.add_argument(
        "--surfaces", nargs="+", default=DEFAULT_SURFACES,
        metavar="SURFACE",
        help="Surfaces to export (default: pial white). Any FreeSurfer surface name is accepted.",
    )
    sel.add_argument(
        "--cortical-regions", nargs="+", default=None, metavar="REGION",
        help=(
            "Cortical lobe regions to export. "
            "Default: all regions defined in region_labels.json. "
            "Available: frontal motor prefrontal parietal temporal occipital"
        ),
    )
    sel.add_argument(
        "--aseg-regions", nargs="+", default=None, metavar="REGION",
        help=(
            "ASeg subcortical regions to export. "
            "Default: curated set of 16 structures. "
            "Use --list-aseg-regions to see all available names."
        ),
    )
    sel.add_argument(
        "--all-aseg", action="store_true",
        help="Export every region in aseg_labels.json (overrides --aseg-regions).",
    )

    # -- Skip flags -----------------------------------------------------------
    skip = parser.add_argument_group("skip flags")
    skip.add_argument("--no-surfaces", action="store_true",
                      help="Skip whole-brain surface (pial/white) export.")
    skip.add_argument("--no-cortical", action="store_true",
                      help="Skip cortical parcellation region export.")
    skip.add_argument("--no-aseg", action="store_true",
                      help="Skip subcortical ASeg region export.")

    # -- Hemisphere handling --------------------------------------------------
    hemi = parser.add_argument_group("hemisphere handling")
    hemi.add_argument(
        "--split-hemispheres", action="store_true",
        help=(
            "Write bilateral structures as separate lh.* and rh.* files "
            "instead of a single combined file."
        ),
    )

    # -- Mesh parameters ------------------------------------------------------
    mesh = parser.add_argument_group("mesh parameters")
    mesh.add_argument(
        "--smooth", type=int, default=15, metavar="N",
        help="Laplacian smoothing iterations for ASeg volume meshes (default: 15, 0 = none).",
    )
    mesh.add_argument(
        "--rotation", type=float, default=90.0, metavar="DEG",
        help="X-axis rotation applied to all output meshes in degrees (default: 90).",
    )
    mesh.add_argument(
        "--scale", type=float, default=0.1, metavar="FACTOR",
        help="Uniform scale factor applied to all output meshes (default: 0.1).",
    )
    mesh.add_argument(
        "--translation", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="XYZ translation applied to all output meshes (default: 0 0 0).",
    )

    # -- Misc -----------------------------------------------------------------
    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--lut-file", metavar="LUT_FILE",
        help="Path to FreeSurferColorLUT.txt file to use instead of default aseg_labels.json.",
    )
    misc.add_argument(
        "--list-aseg-regions", action="store_true",
        help="Print all available ASeg region names and exit.",
    )
    misc.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress information.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load resource files
    if args.lut_file:
        try:
            aseg_labels_map = load_aseg_labels_from_lut(args.lut_file)
        except FileNotFoundError:
            print(f"Error: LUT file not found: {args.lut_file}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            aseg_labels_map = load_json(ASEG_LABELS_PATH)
        except FileNotFoundError as exc:
            print(f"Error: resource file not found: {exc}", file=sys.stderr)
            sys.exit(1)
    
    try:
        region_labels_map = load_json(REGION_LABELS_PATH)
    except FileNotFoundError as exc:
        print(f"Error: resource file not found: {exc}", file=sys.stderr)
        sys.exit(1)

    # Info-only: list aseg regions
    if args.list_aseg_regions:
        col_w = max(len(k) for k in aseg_labels_map)
        source = args.lut_file if args.lut_file else "aseg_labels.json"
        print(f"Available ASeg regions (from {source}):\n")
        print(f"  {'Region':<{col_w}}  Type       Labels")
        print(f"  {'-'*col_w}  ---------  ------")
        for name, indices in aseg_labels_map.items():
            kind = "bilateral" if isinstance(indices, list) else "midline  "
            print(f"  {name:<{col_w}}  {kind}  {indices}")
        sys.exit(0)

    # Validate input
    if not args.input:
        parser.error("--input / -i is required.")

    recon_dir = str(Path(args.input).resolve())
    if not os.path.isdir(recon_dir):
        print(f"Error: input directory does not exist: {recon_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = str(Path(args.output).resolve())
    else:
        output_dir = os.path.join(recon_dir, "wavefront")

    os.makedirs(output_dir, exist_ok=True)

    if args.verbose:
        print(f"Subject dir : {recon_dir}")
        print(f"Output dir  : {output_dir}")

    # Determine region lists
    cortical_regions = args.cortical_regions or list(region_labels_map.keys())

    if args.all_aseg:
        aseg_regions = list(aseg_labels_map.keys())
    elif args.aseg_regions:
        aseg_regions = args.aseg_regions
    else:
        aseg_regions = DEFAULT_ASEG_REGIONS

    translation = tuple(args.translation)

    # Run pipeline -----------------------------------------------------------

    if not args.no_surfaces:
        if args.verbose:
            print("\n[1/3] Processing whole-brain surfaces ...")
        process_surfaces(
            recon_dir, output_dir, args.surfaces, args.split_hemispheres,
            args.rotation, args.scale, translation, args.verbose,
        )

    if not args.no_cortical:
        if args.verbose:
            print("\n[2/3] Processing cortical parcellation regions ...")
        process_cortical_regions(
            recon_dir, output_dir, region_labels_map, cortical_regions,
            args.split_hemispheres, args.rotation, args.scale, translation, args.verbose,
        )

    if not args.no_aseg:
        if args.verbose:
            print("\n[3/3] Processing subcortical aseg regions ...")
        process_aseg_regions(
            recon_dir, output_dir, aseg_labels_map, aseg_regions,
            args.split_hemispheres, args.smooth,
            args.rotation, args.scale, translation, args.verbose,
        )

    if args.verbose:
        print(f"\nDone. Output written to: {output_dir}")


if __name__ == "__main__":
    main()
