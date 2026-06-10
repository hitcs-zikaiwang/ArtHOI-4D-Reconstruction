import pyrealsense2 as rs2
import numpy as np
from loguru import logger as loguru
from arrgh import arrgh
from PIL import Image
import cv2
import os
import json
import argparse
import imageio
import open3d as o3d
import pickle

def save_intr(color_intrinsics, depth_scale, output_dir):
    cam_info = dict()
    cam_info['im_w'] = color_intrinsics.width
    cam_info['im_h'] = color_intrinsics.height
    cam_info['depth_scale'] = depth_scale
    fx, fy = color_intrinsics.fx, color_intrinsics.fy
    cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
    cam_info['cam_intr'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    
    if output_dir is not None:
        cam_info['id'] = os.path.basename(output_dir)
        cam_config_path = os.path.join(output_dir, 'intrinsics.json')
        with open(cam_config_path, 'w') as f:
            loguru.info(f"Camera info has been saved to: {cam_config_path}.")
            json.dump(cam_info, f, indent=4)
    return cam_info

def save_video(frames, fps, output_name):
    """
    using RGB frames
    """
    try:
        os.makedirs(os.path.dirname(output_name) or '.', exist_ok=True)
        # materialize frames in case an iterator/generator was passed
        frames_list = list(frames)
        if len(frames_list) == 0:
            loguru.warning("No frames to save, skipping video write.")
            return

        with imageio.get_writer(output_name, fps=fps) as writer:
            for frm in frames_list:
                f = frm
                if isinstance(f, np.ndarray):
                    # normalize float frames to uint8 [0,255]
                    if f.dtype in (np.float32, np.float64):
                        f = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
                    elif f.dtype != np.uint8:
                        f = f.astype(np.uint8)
                writer.append_data(f)

        loguru.info(f"Saved video to: {output_name}")
    except Exception:
        loguru.exception(f"Failed to save video to: {output_name}")


def main(args):
    bag_path = args.bag
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # load bag
    pipeline = rs2.pipeline()
    config = rs2.config()
    # config.enable_device_from_file(bag_path)
    config.enable_device_from_file(bag_path, repeat_playback=False)

    # fixed settings for ArtHOI captures
    config.enable_stream(rs2.stream.depth, rs2.format.z16, 30)
    config.enable_stream(rs2.stream.color, rs2.format.rgb8, 30)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback() # get playback device
    playback.set_real_time(False) # disable real-time playback
    
    color_profile = rs2.video_stream_profile(profile.get_stream(rs2.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    loguru.info(f'Color Intrinsics: {color_intrinsics}')
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    loguru.info(f"Depth Scale is: {depth_scale}")
    
    rs_cam_intr = save_intr(color_intrinsics, depth_scale, output_dir)
    
    align_to = rs2.stream.color
    align = rs2.align(align_to)

    colorizer = rs2.colorizer()
    
    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    
    frame_no = 0
    rgb_frames = []
    depth_frames_metric = []
    depth_frames_raw = []
    depth_vis_frames = []
    while True:
        try:
            raw_frames = pipeline.wait_for_frames()
        except RuntimeError as e:
            loguru.error(f"Error occurred while waiting for frames: {e}")
            cv2.destroyAllWindows()
            break
        
        aligned_frames = align.process(raw_frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            loguru.warning('No frame received, skipping...')
            continue
        depth_color_frame = colorizer.colorize(depth_frame)
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_color_frame.get_data())
        depth_image_raw = np.asanyarray(depth_frame.get_data())
        loguru.info(f'Loaded frame of shape: C{color_image.shape} D{depth_image.shape}')
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # img = Image.fromarray(color_image)
        # img.show()
        """
        pc = rs2.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        points.export_to_ply(f"{output_dir}/{frame_no}.ply", color_frame)
        """
        """
        # # let's show pointcloud
        depth_im_metric = depth_image_raw.astype(np.float32) / depth_scale
        # to [H, W, 3]
        depth_im_metric = np.stack([depth_im_metric] * 3, axis=-1)
        depth_im_o3d = o3d.geometry.Image(depth_im_metric)
        # depth_im_o3d = o3d.geometry.Image(depth_image_raw)
        color_im_o3d = o3d.geometry.Image(color_image)
        cam_intr_o3d = o3d.camera.PinholeCameraIntrinsic()
        cam_intr_o3d.intrinsic_matrix = np.array(rs_cam_intr['cam_intr'])
        depth_trunc = 4.0
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_im_o3d, depth_im_o3d, depth_scale=1,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr_o3d)
        o3d.visualization.draw_geometries_with_editing([pcd])
        input(f'Press Enter to continue...')
        """
        
        def mouse_callback(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                d = depth_image_raw[y, x]
                d_meter = d.astype(np.float32) * depth_scale
                loguru.debug(f"({x}, {y}), depth: {d_meter}m")

        cv2.imshow("Depth Stream", depth_image)
        cv2.setMouseCallback("Depth Stream", mouse_callback)
        cv2.imshow("Color Stream", color_image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        rgb_frames.append(color_image.copy())
        depth_vis_frames.append(depth_image.copy())
        depth_frames_raw.append(depth_image_raw.copy())
        depth_frames_metric.append(depth_image_raw.astype(np.float32) * depth_scale)
        frame_no += 1

    # gracefully release resources
    pipeline.stop()
    
    # cuts
    loguru.critical(f'Total {frame_no} frames loaded. Will save {args.from_frame} to {args.to_frame} cuts')
    rgb_frames = np.array(rgb_frames)[args.from_frame: args.to_frame]
    depth_vis_frames = np.array(depth_vis_frames)[args.from_frame: args.to_frame]


    # save rgb frames
    depth_vis_frames = depth_vis_frames[:, :, :, ::-1] # to RGB for better visual 
    os.makedirs(f'{output_dir}/rgb_frames', exist_ok=True)
    os.makedirs(f'{output_dir}/depth_frames', exist_ok=True)
    for fid, (rgb, dv) in enumerate(zip(rgb_frames, depth_vis_frames)):
        im_pil = Image.fromarray(rgb)
        im_pil.save(f"{output_dir}/rgb_frames/rgb_{fid}.png")
        dv_pil = Image.fromarray(dv)
        dv_pil.save(f"{output_dir}/depth_frames/depth_{fid}.png")
    save_video(rgb_frames, fps=30, output_name=os.path.join(output_dir, 'rgb.mp4'))
    save_video(depth_vis_frames, fps=30, output_name=os.path.join(output_dir, 'depth_vis.mp4'))
    
    # save depth data
    if args.save_depth:
        depth_raw_np = np.array(depth_frames_raw)[args.from_frame: args.to_frame]
        depth_metric_np = np.array(depth_frames_metric)[args.from_frame: args.to_frame]
        np.save(os.path.join(output_dir, 'depth_raw.npy'), depth_raw_np)
        np.save(os.path.join(output_dir, 'depth_metric.npy'), depth_metric_np)
    
    metadata = {}
    metadata['frame_count'] = frame_no
    metadata['from_frame'] = args.from_frame
    metadata['to_frame'] = args.to_frame
    metadata['save_depth'] = args.save_depth
    metadata['bag_path'] = bag_path
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == '__main__':
    from datetime import datetime
    parser = argparse.ArgumentParser(description='RealSense Recording Pose Processing')
    parser.add_argument('-b', '--bag', required=True, help='path to the bag file')
    parser.add_argument('--exp_name', default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='experiment name')
    parser.add_argument('-o', '--output', default=f'./outputs/realsense/', help='path to the output directory')
    parser.add_argument('-f', '--from_frame', type=int, default=0, help='frame number to start save from')
    parser.add_argument('-t', '--to_frame', type=int, default=-1, help='frame number to stop save at')
    parser.add_argument('--save-depth', action='store_true', help='whether to save depth data')
    args = parser.parse_args()
    args.output = os.path.join(args.output, args.exp_name)
    if args.to_frame != -1:
        args.to_frame = args.to_frame + 1
    main(args)