# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import os
import open3d as o3d
import numpy as np
import torch
import nvdiffrast.torch as dr
from loguru import logger as loguru

import sys
__cur_path__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{__cur_path__}/../third_party")
from foundationpose.Utils import *
from foundationpose.datareader import *
from foundationpose.learning.training.predict_score import *
from foundationpose.learning.training.predict_pose_refine import *

import hrse.utils.plot.asr_plotter as asr_plotter
import hrse.utils.tfs.backproj as backproj
import hrse.utils.image_utils as image_utils

class ASR:
    def __init__(self, symmetry_tfs=None, mesh=None, scorer: ScorePredictor = ScorePredictor(),
                 refiner: PoseRefinePredictor = PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(), debug=0,
                 debug_dir='./debug/'):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir

        self.refiner_dir = os.path.join(self.debug_dir,"refine")
        self.scorer_dir = os.path.join(self.debug_dir,"score")
        if self.debug != 0:
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(self.refiner_dir, exist_ok=True)
            os.makedirs(self.scorer_dir, exist_ok=True)

        self.reset_object(mesh=mesh, symmetry_tfs=symmetry_tfs)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh

    def reset_object(self, mesh=None, symmetry_tfs=None):
        min_xyz = mesh.vertices.min(axis=0)
        max_xyz = mesh.vertices.max(axis=0)
        self.model_center = (min_xyz+max_xyz)/2
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            rgb_colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb_colors)

        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and mesh.visual.material.image is not None:
            img = copy.deepcopy(np.array(mesh.visual.material.image.convert('RGB')))

            uv = copy.deepcopy(mesh.visual.uv)
            uv[:, 1] = 1 - uv[:, 1]

            uv_pixels_y = np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1)
            uv_pixels_x = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)

            vertex_colors = img[uv_pixels_y, uv_pixels_x].astype(float) / 255.0

            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        mesh_o3d.compute_vertex_normals()
        self.mesh_o3d = mesh_o3d
        # model_pts = mesh.vertices
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        # logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        # pcd = toOpen3dCloud(model_pts, normals=model_normals)
        # pcd = pcd.voxel_down_sample(self.vox_size)
        # self.max_xyz = np.asarray(pcd.points).max(axis=0)
        # self.min_xyz = np.asarray(pcd.points).min(axis=0)
        # self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        # self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
        # logging.info(f'self.pts:{self.pts.shape}')
        # self.mesh_path = None
        self.mesh = mesh
        # if self.mesh is not None:
        #     self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
        #     self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        # logging.info("reset done")

    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3, 3] = -torch.FloatTensor(self.model_center.copy()).cuda()
        return tf_to_center

    def to_device(self, s='cuda:0'):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                # logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            # logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        # logging.info(f'cam_in_obs:{cam_in_obs.shape}')
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        # logging.info(f"rot_grid:{rot_grid.shape}")
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        # logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        # logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None,initial_center=False):
        '''
        @scene_pts: torch tensor (N,3)
        '''
        ob_in_cams = self.rot_grid.clone()
        if initial_center:
            center = self.guess_translation_bounding_box(depth=depth, mask=mask, K=K)
        else:
            center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation_bounding_box(self, depth, mask, K):
        xyz_map = depth2xyzmap(depth, K)

        xyz_map[mask == False] = 0
        points = xyz_map[mask].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(np.array(points).shape[0] * 0.01), std_ratio=2.0)

        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
        pcd_clean.orient_normals_consistent_tangent_plane(k=1000)
        obb_pcd_clean = pcd_clean.get_oriented_bounding_box()

        center = obb_pcd_clean.get_center()

        # if self.debug >= 2:
        #     o3d.io.write_point_cloud(f'{self.debug_dir}/points.ply', pcd)
        #     # Save bounding box for visualization
        #     bbox_points = np.asarray(obb.get_box_points())
        #     bbox_pcd = o3d.geometry.PointCloud()
        #     bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)
        #     o3d.io.write_point_cloud(f'{self.debug_dir}/bbox.ply', bbox_pcd)

        return np.asarray(center)

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            # logging.info(f'mask is all zero')
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            # logging.info(f"valid is empty")
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        return center.reshape(3)

    def register_asr(self, K, rgb, depth, ob_mask, asr_iter=20, asr_seed=0,
                     ob_id=None, glctx=None, iteration=5, name=None, 
                     refinement=True, axis_align=False,coarse_est=True):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        '''
        set_seed(asr_seed)

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = glctx

        depth_pcl, _ = backproj.metric_depth_to_pointcloud(
            torch.from_numpy(depth).to('cuda'), K, np.eye(4), # FIXME remove hardcoded cuda device
            rgb=None,
            mask=image_utils.erode_mask(torch.from_numpy(ob_mask).to('cuda'), kernel_size=5),
            return_o3d=False,
            # fast_impl=True,  ## will lower precision, but way more faster
        )
        depth_pcl = depth_pcl.cpu().numpy()

        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        xyz_map = depth2xyzmap(depth, K)
        xyz_map[ob_mask == False] = 0
        loguru.info('eroded and filtered depth')
        
        """
        points = xyz_map[ob_mask].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.529, 0.808, 0.922], (len(pcd.points), 1)))  # 모든 point를 파란색으로

        loguru.info("removing outliers...")
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=int(points.shape[0] * 0.01), std_ratio=2.0)
        """


        loguru.info(f"executing coarse est...")
        if coarse_est:
            mesh_pcd = copy.deepcopy(self.mesh_o3d)
            pts_mesh = np.asarray(mesh_pcd.vertices)
            bbox_diag_depth = np.linalg.norm(
                depth_pcl.max(axis=0) - depth_pcl.min(axis=0)
            )
            bbox_diag_mesh = np.linalg.norm(
                pts_mesh.max(axis=0) - pts_mesh.min(axis=0)
            )
            scale_coarse = bbox_diag_depth / bbox_diag_mesh
            
            # pcd_ = mesh_pcd.sample_points_uniformly(number_of_points=100000)
            # cl, _ = pcd_.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # obb_mesh = cl.get_oriented_bounding_box()
            # obb_mesh.color = (0, 0, 1)
            # if axis_align:
            #     pcd_clean.rotate(obb_mesh.R @ obb_pcd_clean.R.T, center=obb_pcd_clean.center)
            # else:
            #     pass
            # obb_pcd_clean = pcd_clean.get_oriented_bounding_box(robust=True)
            # obb_pcd_clean.color = (0, 1, 0)
            # extent_pcd_clean = obb_pcd_clean.extent
            # extent_mesh = obb_mesh.extent

            # ratio, best_perm, best_iou = find_best_ratio_combination(extent_pcd_clean, extent_mesh, obb_pcd_clean, obb_mesh)
            # mesh_pcd.scale(ratio[1], center=obb_mesh.center) # y axis (height)
            obb_mesh = mesh_pcd.get_oriented_bounding_box(robust=True)
            obb_mesh.color = (1, 0, 0)
            mesh_pcd.scale(scale_coarse, center=obb_mesh.center)

            mesh = copy.deepcopy(self.mesh)
            mesh.vertices = np.asarray(mesh_pcd.vertices)
            self.reset_object(mesh=mesh, symmetry_tfs=self.symmetry_tfs)
            # self.mesh.export(os.path.join(self.debug_dir, f'refine_init_mesh_{name}.obj'))
            # print(f"coarse est ratio: {ratio}, perm: {best_perm}, iou: {best_iou}")
            print(f"coarse est ratio: {scale_coarse}")

        valid_indices = np.argwhere(ob_mask)  # Get (h, w) coordinates where ob_mask is True
        # selected_indices = valid_indices[ind]  # Use the indices from the outlier filtering to get (h, w)
        selected_indices = valid_indices#[ind]  # Use the indices from the outlier filtering to get (h, w)

        depth_mask = np.zeros((xyz_map.shape[:2]), dtype=bool)
        depth_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
        xyz_map[depth_mask == False] = 0

        depth = xyz_map[..., -1]

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            # logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

        xyz_map = depth2xyzmap(depth, K)

        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)

        # if vis is not None:
        #     imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_1_consider_pose_{name}.png', vis)

        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)
        # if vis is not None:
        #     imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_1_consider_pose_{name}.png', vis)

        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]
        # best_pose = poses[0].data.cpu().numpy()

        best_pose = poses[0].data.cpu().numpy()
        best_score = scores[0].data.cpu().numpy()
        best_scale = np.array([1.0, 1.0, 1.0])
        best_iou = 0.0
        if refinement:
            loguru.info(f"Final random refining pose for {name}")
            sample_delta = 0.03 # 3% initial
            pre_delta = 0.0
            i_upd, d_upd = 0, 0
            ITER_N = asr_iter
            for _i in range(ITER_N):
                loguru.info(f"\nRandScale {_i}/{ITER_N-1}\n"
                            f"delta={sample_delta}, bIoU={best_iou:.4f}\tbest_iter={i_upd}, deltaUpd_iter={d_upd}")
                # Define the number of samples
                num_samples = 64
                # num_samples = 252

                # Define the scaling ratios for each axis
                # ratio_i, ratio_l = 0.6, 1.4
                ratio_i, ratio_l = [1 - sample_delta, 1 + sample_delta]
                ratios = {'x': (ratio_i, ratio_l), 'y': (ratio_i, ratio_l), 'z': (ratio_i, ratio_l)}

                # Generate random scaling values for each axis
                samples = {axis: np.random.uniform(*ratio, num_samples) for axis, ratio in ratios.items()}

                # update samples so z scale is either aligned with x or y
                # for i in range(num_samples):
                #     if np.random.rand() < 0.5:
                #         samples['z'][i] = samples['x'][i]
                #     else:
                #         samples['z'][i] = samples['y'][i]
                
                """
                ratio_i_range = (1 - sample_delta, 1 - pre_delta)
                ratio_l_range = (1 + pre_delta, 1 + sample_delta)
                num_low = num_samples // 2
                num_high = num_samples - num_low
                # draw samples for each axis
                samples = {}
                for axis in ['x', 'y', 'z']:
                    ris = np.random.uniform(*ratio_i_range, num_low)
                    rls = np.random.uniform(*ratio_l_range, num_high)
                    samples[axis] = np.concatenate([ris, rls])
                    np.random.shuffle(samples[axis])
                """
                # Create scaling matrices
                scaling_matrices = np.array([np.diag([samples['x'][i], samples['y'][i], samples['z'][i], 1]) for i in range(num_samples)])
                # This creates 252 4x4 diagonal matrices, each representing a scaling transformation

                # Generate final transformation matrices
                final_transforms = np.einsum('ij,njk->nik', best_pose, scaling_matrices)
                # This applies the best_pose transformation to each scaling matrix [252, 4, 4]

                rescale_poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb,
                                                          depth=depth,
                                                          K=K,
                                                          ob_in_cams=final_transforms, normal_map=normal_map,
                                                          xyz_map=xyz_map,
                                                          glctx=self.glctx, mesh_diameter=self.diameter,
                                                          iteration=iteration,
                                                          get_vis=self.debug >= 2)
                # if vis is not None:
                #     imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_3_consider_size.png', vis)

                rescale_scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                                          ob_in_cams=rescale_poses.data.cpu().numpy(),
                                                          normal_map=normal_map,
                                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                                          mesh_diameter=self.diameter,
                                                          get_vis=self.debug >= 2)
                # if vis is not None:
                #     imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_3_consider_size.png', vis)
                # print('')

                # combine_scores = scores + rescale_scores
                # combine_ids = torch.as_tensor(combine_scores).argsort(descending=True)
                rescale_ids = torch.as_tensor(rescale_scores).argsort(descending=True)

                # rescale_scores = rescale_scores[rescale_ids]
                rescale_poses = rescale_poses[rescale_ids]
                scaling_matrices = scaling_matrices[rescale_ids.detach().cpu().numpy()]
                
                curr_best_scale = np.array([scaling_matrices[0][0, 0], scaling_matrices[0][1, 1], scaling_matrices[0][2, 2]])
                
                # calc IoU
                scaled_verts = self.mesh.vertices * curr_best_scale
                rend_pose = rescale_poses[0] @ self.compute_scaled_model_center_tf(scaled_verts)
                Liou, rend = self.render_2dIoU(
                    best_pose=rend_pose.data.cpu().numpy(), 
                    mesh_v=scaled_verts, 
                    mesh_f=self.mesh.faces, 
                    mesh_c=self.mesh.visual.vertex_colors[:, :3].astype(float),
                    color=rgb, 
                    mask=ob_mask, 
                    K=K,
                    save_path=f"{self.debug_dir}/rescaled_render_{name}_iter{_i}.png" if self.debug >=2 else None
                )
                
                loguru.info(f"\ncurrent iter best scale={curr_best_scale}, IoU={Liou:.4f}, overall_best_IoU={best_iou:.4f}")
                if Liou > best_iou:
                    best_iou = Liou
                    best_pose = rescale_poses[0].data.cpu().numpy()
                    best_score = rescale_scores[0].data.cpu().numpy()
                    best_scale = curr_best_scale
                    i_upd = _i
                    # loguru.info(f"Updated best scale to {best_scale}, bIoU={best_iou:.4f}")
                if i_upd * 2 < (_i - 1) and (_i - d_upd) >= 4:
                    pre_delta = sample_delta
                    sample_delta *= 2.0
                    d_upd = _i
                
            print(f"best scale={best_scale}")

            self.mesh.vertices = self.mesh.vertices * best_scale
            self.reset_object(mesh=self.mesh, symmetry_tfs=self.symmetry_tfs)
            self.mesh.export(os.path.join(self.debug_dir, f'final_mesh_{name}.obj'))

            poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              xyz_map=xyz_map,
                                              glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                              get_vis=self.debug >= 2)
            # if vis is not None:
            #     imageio.imwrite(f'{self.refiner_dir}/vis_refiner_stage_4_rerun_pose.png', vis)

            scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                              ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                              mesh_tensors=self.mesh_tensors, glctx=self.glctx,
                                              mesh_diameter=self.diameter,
                                              get_vis=self.debug >= 2)
            # if vis is not None:
            #     imageio.imwrite(f'{self.scorer_dir}/vis_score_stage_4_rerun_pose.png', vis)

            ids = torch.as_tensor(scores).argsort(descending=True)
            # logging.info(f'sort ids:{ids}')
            scores = scores[ids]
            poses = poses[ids]
            # update pose
            rend_pose = poses[0] @ self.get_tf_to_centered_mesh()
            Liou, rend = self.render_2dIoU(
                best_pose=rend_pose.data.cpu().numpy(), 
                mesh_v=self.mesh.vertices, 
                mesh_f=self.mesh.faces, 
                mesh_c=self.mesh.visual.vertex_colors[:, :3].astype(float),
                color=rgb, 
                mask=ob_mask, 
                K=K,
                save_path=None
            )
            loguru.info(f"Last rerun pose IoU: {Liou:.4f}")
            if Liou > best_iou:
                best_iou = Liou
                best_pose = poses[0].data.cpu().numpy()
                best_score = scores[0].data.cpu().numpy()
        # logging.info(f'sorted scores:{scores}')
        
        # best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        best_pose = best_pose @ self.get_tf_to_centered_mesh().cpu().numpy()

        self.pose_last = poses[0]
        self.best_id = ids[0]
        self.poses = poses
        self.scores = scores
        best_score = scores[0]
        
        Liou, rend = self.render_2dIoU(
            best_pose=best_pose, 
            mesh_v=self.mesh.vertices, 
            mesh_f=self.mesh.faces, 
            mesh_c=self.mesh.visual.vertex_colors[:, :3].astype(float),
            color=rgb, 
            mask=ob_mask, 
            K=K,
            save_path=f"{self.debug_dir}/best_render_{name}.png" if self.debug >=2 else None
        )
        loguru.info(f"Final return pose IoU: {Liou:.4f}, Final Best IoU: {best_iou:.4f}")
        assert np.abs(Liou - best_iou) < 1e-6, 'error in IoU calc'
        
        return best_pose, best_score, best_scale, best_iou, rend

    def render_2dIoU(self, best_pose, mesh_v, mesh_f, mesh_c, color, mask, K, save_path=None):
        """
        should all be nparray
        """
        if np.max(color) > 2.0:
            color = color.astype(np.float32) / 255.0
        rend = asr_plotter.render_deform_oneframe(
            color, mask, K, 
            mesh_v=mesh_v, mesh_f=mesh_f, mesh_c=mesh_c,
            tf_mat=best_pose,
            save=save_path
        )
        iou = image_utils.mask_IoU(
            rend['mask'], mask
        )
        loguru.debug(f"PyT3D rendered IoU: {iou:0.4f}")
        return iou, rend

    def compute_scaled_model_center_tf(self, verts):
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        model_center = (min_xyz+max_xyz)/2
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3, 3] = -torch.FloatTensor(model_center).cuda()
        return tf_to_center

