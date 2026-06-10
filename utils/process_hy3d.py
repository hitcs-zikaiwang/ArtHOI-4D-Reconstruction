import trimesh
import numpy as np
import os
import argparse
import pymeshlab
from loguru import logger as loguru

def process_glb_to_vertcolor_obj(glb_path, output_path):
    """
    Convert GLB file to OBJ with vertex colors, without saving texture.
    
    Args:
        glb_path (str): Path to the input GLB file.
        output_path (str): Path to save the output OBJ file.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(glb_path)
    ms.transfer_texture_to_color_per_vertex()
    # delete material
    ms.save_current_mesh(
        output_path, 
        save_vertex_color=True,
        save_textures=False,
        save_face_color=False,
        save_vertex_normal=False,
        save_vertex_coord=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh processing')
    parser.add_argument('--seq-path', type=str, default=None)
    parser.add_argument('--mesh_path', default=None)
    args = parser.parse_args()
    
    if args.mesh_path is not None:
        mesh_path = args.mesh_path
    elif args.seq_path is not None:
        mesh_path = os.path.join(args.seq_path, 'build', 'mesh')
    else:
        raise ValueError("Either --seq-path or --mesh_path must be provided.")
    # list all glb
    glb_files = [f for f in os.listdir(mesh_path) if f.endswith('.glb')]
    for glb_file in glb_files:
        process_glb_to_vertcolor_obj(
            os.path.join(mesh_path, glb_file),
            os.path.join(mesh_path, glb_file.replace('.glb', '.obj'))
        )
    loguru.info(f"Processed {len(glb_files)} GLB files to OBJ with vertex colors in {mesh_path}")