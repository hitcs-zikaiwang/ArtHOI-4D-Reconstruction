import numpy as np
import torch


def main():
    artipart_dump_path = input(f'input arti partdata path: ')
    
    
    part_data = np.load(artipart_dump_path, allow_pickle=True).item()
    vmasks = part_data['vertex_masks']  # list of (frame_cnt,) bool tensor
    verts_cnt = vmasks[0].shape[0]
    vmask_large = -1 * np.ones(verts_cnt, dtype=int)  # -1 means not covered
    
    id_map = []
    for f in range(len(vmasks)):
        mval = input(f'part {f} will map to group: ')
        id_map.append(int(mval))
    id_map = np.array(id_map, dtype=int)
    print(f"ID map: {id_map}")
    
    # set to the first frame that covers this vertex
    for f, vmask in enumerate(vmasks):
        vmask_np = vmask.cpu().numpy()
        vmask_large[(vmask_np) & (vmask_large == -1)] = id_map[f]
    
    not_covered = np.sum(vmask_large == -1)
    if not_covered > 0:
        print(f"{not_covered} vertices are not covered by any frame!")
    
    out_path = input(f'input output npy path: ')
    np.save(f'{out_path}/part_vertex_mask.npy', vmask_large)
    print(f"Saved initial mask to {out_path}")


if __name__ == '__main__':
    main()