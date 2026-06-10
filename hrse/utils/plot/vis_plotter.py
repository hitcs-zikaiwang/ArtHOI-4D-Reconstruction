import torch
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # faster non-interactive backend for matplotlib
# monkey-patch, see: https://github.com/isl-org/Open3D/issues/1715
# matplotlib.use('Cairo') # or QtAgg
# matplotlib.use('TkAgg')  # or QtAgg, TkAgg, etc.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw

from hrse.utils.image_utils import erode_mask, dilate_mask, colorize_depth

def to_numpy(d) -> np.ndarray:
    if isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    else:
        return d

def PIL_show(img):
    if isinstance(img, torch.Tensor):
        img = img.clone().detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported type: {type(img)}")
    if img.max() <= 2.0:
        img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    if img.ndim == 2 or img.shape[-1] == 1:
        pimg = Image.fromarray(img.squeeze(), mode='L')
    else:
        pimg = Image.fromarray(img.astype(np.uint8))
    pimg.show()

def rgb_overlap(im_top, im_bot, top_a=1.0, bot_a=0.35, bkg=(0.0, 0.0, 0.0)):
    """Overlay two RGB images with alpha blending.
    Args:
        im_top: HxWx3, top image
        im_bot: HxWx3, bottom image
        top_a: float, alpha for top image
        bot_a: float, alpha for bottom image
        bkg: tuple of 3 floats, background color for areas where both images are transparent
    Returns:
        HxWx3, blended image
    """
    im_top = to_numpy(im_top).clip(0, 1)
    im_bot = to_numpy(im_bot).clip(0, 1)
    h, w = im_top.shape[:2]
    blended = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        blended[..., c] = (
            im_top[..., c] * top_a + 
            im_bot[..., c] * bot_a + 
            bkg[c] * (1 - top_a) * (1 - bot_a)
        )
    blended = np.clip(blended, 0, 1)
    return blended

def plot_rgb_masks(
    gt_rgb, gt_mask, 
    pred_rgb, pred_mask, 
    save=None,
    ret_pic=False,
):
    gt_rgb = to_numpy(gt_rgb).clip(0, 1)
    gt_mask = to_numpy(gt_mask).clip(0, 1)
    pred_rgb = to_numpy(pred_rgb).clip(0, 1)
    pred_mask = to_numpy(pred_mask).clip(0, 1)
    
    fig = plt.figure(figsize=(12, 9))

    # Plot predicted RGB and mask
    plt.subplot(3, 2, 1)
    plt.imshow(pred_rgb)
    plt.title('Predicted Rendered')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    # Plot ground truth RGB and mask
    plt.subplot(3, 2, 3)
    plt.imshow(gt_rgb)
    plt.title('Ground Truth RGB')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Add RGB overlapped visualization
    plt.subplot(3, 2, 5)
    rgb_black = np.zeros_like(pred_rgb)
    rgb_black[pred_mask.astype(bool)] = pred_rgb[pred_mask.astype(bool)]
    overlapped_rgb = rgb_overlap(rgb_black, gt_rgb, top_a=0.60, bot_a=0.30)
    plt.imshow(overlapped_rgb)
    plt.title('RGB Overlapped')
    plt.axis('off')
    
    # Add mask overlapped visualization with different colors
    plt.subplot(3, 2, 6)
    # Create RGB mask visualization with different colors
    mask_vis = np.zeros((*gt_mask.shape, 3))
    mask_vis[..., 0] = gt_mask    # Red channel for ground truth
    mask_vis[..., 1] = pred_mask  # Green channel for prediction
    plt.imshow(mask_vis)
    plt.title('Mask Overlapped')
    plt.axis('off')
    
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    elif ret_pic:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert("RGB")
        return img
    else:
        plt.close(fig)


def plot_depth_masks(
    gt_depth, gt_mask, 
    pred_depth, pred_mask, 
    erode=5,
    save=None,
    ret_pic=False,
):
    gt_mask_e = erode_mask(gt_mask, kernel_size=erode)
    pred_mask_e = erode_mask(pred_mask, kernel_size=erode)
    gt_depth = to_numpy(gt_depth)
    gt_mask = to_numpy(gt_mask).astype(bool)
    pred_depth = to_numpy(pred_depth)
    pred_mask = to_numpy(pred_mask).astype(bool)
    gt_mask_e = to_numpy(gt_mask_e).astype(bool)
    pred_mask_e = to_numpy(pred_mask_e).astype(bool)
    
    if gt_mask_e.sum() == 0 or pred_mask_e.sum() == 0:
        gt_dmax = gt_dmin = 0
    else:
        gt_dmax = np.max(gt_depth[gt_mask_e])
        gt_dmin = np.min(gt_depth[gt_mask_e])
    if pred_mask_e.sum() == 0 or pred_mask_e.sum() == 0:
        pred_dmax = pred_dmin = 0
    else:
        pred_dmax = np.max(pred_depth[pred_mask_e])
        pred_dmin = np.min(pred_depth[pred_mask_e])
    
    # Convert depth maps to RGB visualization
    gt_depth = np.where(gt_mask_e, gt_depth, 0)
    pred_depth = np.where(pred_mask_e, pred_depth, 0)
    gt_depth_rgb = colorize_depth(gt_depth, valid_mask=gt_mask_e).clip(0, 1)
    pred_depth_rgb = colorize_depth(pred_depth, valid_mask=pred_mask_e).clip(0, 1)
    
    fig = plt.figure(figsize=(12, 9))

    # Plot predicted depth and mask
    plt.subplot(3, 2, 1)
    plt.imshow(pred_depth_rgb)
    plt.title(f'Predicted Depth [m={pred_dmin:.3f} M={pred_dmax:.3f}]')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    # Plot ground truth depth and mask
    plt.subplot(3, 2, 3)
    plt.imshow(gt_depth_rgb)
    plt.title(f'GT Depth [m={gt_dmin:.3f} M={gt_dmax:.3f}]')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Add depth overlapped visualization
    plt.subplot(3, 2, 5)
    # Blend the two RGB depth maps
    blended_depth = 0.7 * gt_depth_rgb + 0.5 * pred_depth_rgb
    # Normalize to ensure values are in valid range
    blended_depth = np.clip(blended_depth, 0, 1)
    plt.imshow(blended_depth)
    plt.title('Depth Overlapped')
    plt.axis('off')
    
    # Add mask overlapped visualization with different colors
    plt.subplot(3, 2, 6)
    # Create RGB mask visualization with different colors
    mask_vis = np.zeros((*gt_mask.shape, 3))
    mask_vis[..., 0] = gt_mask   # Red channel for ground truth
    mask_vis[..., 1] = pred_mask # Green channel for prediction
    plt.imshow(mask_vis)
    plt.title('Mask Overlapped')
    plt.axis('off')
    
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    elif ret_pic:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert("RGB")
        return img
    else:
        # plt.show()  # Removed slow display.
        plt.close(fig)

def plot_kp_onto_image(
    keypoints, # N, 2
    img, # mask or rgb, in numpy
    radius=4,
    color=(255, 0, 0), # default blue
    return_matplot=False, # return matplotlib figure
    return_pil=False, # return PIL Image
    output_path=None,
):
    keypoints = to_numpy(keypoints)
    img = to_numpy(img)
    if img.max() <= 2.0:
        img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    if img.ndim == 2 or img.shape[-1] == 1:
        img = np.stack([img.squeeze()]*3, axis=-1)
    fig, ax = plt.subplots()
    ax.imshow(img.astype(np.uint8))
    ax.axis('off')
    for kp in keypoints:
        circle = plt.Circle((kp[0], kp[1]), radius, color=np.array(color)/255.0, fill=True)
        ax.add_patch(circle)
    plt.tight_layout()
    
    if output_path is not None:
        if not output_path.endswith('.png'):
            raise ValueError("Only .png format ending is supported for output_path")
        fig.savefig(output_path, bbox_inches='tight')
        
    if return_matplot:
        return fig
    elif return_pil:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert("RGB")
        return img
    
    plt.close(fig)

def plot_rgb_masks_fast(
    gt_rgb, gt_mask, 
    pred_rgb, pred_mask, 
    save=None,
    ret_pic=False,
):
    """Fast OpenCV/PIL version, roughly 10-50x faster than matplotlib."""
    gt_rgb = to_numpy(gt_rgb).clip(0, 1)
    gt_mask = to_numpy(gt_mask).clip(0, 1)
    pred_rgb = to_numpy(pred_rgb).clip(0, 1)
    pred_mask = to_numpy(pred_mask).clip(0, 1)
    
    # Convert to uint8
    gt_rgb = (gt_rgb * 255).astype(np.uint8)
    pred_rgb = (pred_rgb * 255).astype(np.uint8)
    gt_mask = (gt_mask * 255).astype(np.uint8)
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    # Ensure RGB format
    if gt_rgb.ndim == 2:
        gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_GRAY2RGB)
    if pred_rgb.ndim == 2:
        pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_GRAY2RGB)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[:, :, 0]
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]
    
    # Convert masks to RGB
    gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)
    pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
    
    # RGB overlapped
    rgb_overlap = cv2.addWeighted(gt_rgb, 0.7, pred_rgb, 0.5, 0)
    
    # Mask overlapped (red=GT, green=pred)
    mask_overlap = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    mask_overlap[..., 0] = gt_mask    # Red for GT
    mask_overlap[..., 1] = pred_mask  # Green for pred
    
    # Create grid layout
    h, w = gt_rgb.shape[:2]
    grid = np.ones((h * 3, w * 2, 3), dtype=np.uint8) * 255
    
    # Place images
    grid[0:h, 0:w] = pred_rgb
    grid[0:h, w:w*2] = pred_mask_rgb
    grid[h:h*2, 0:w] = gt_rgb
    grid[h:h*2, w:w*2] = gt_mask_rgb
    grid[h*2:h*3, 0:w] = rgb_overlap
    grid[h*2:h*3, w:w*2] = mask_overlap
    
    # Add titles using PIL
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    titles = [
        (10, 5, 'Predicted Rendered'),
        (w + 10, 5, 'Predicted Mask'),
        (10, h + 5, 'Ground Truth RGB'),
        (w + 10, h + 5, 'Ground Truth Mask'),
        (10, h*2 + 5, 'RGB Overlapped'),
        (w + 10, h*2 + 5, 'Mask Overlapped'),
    ]
    
    for x, y, title in titles:
        draw.text((x, y), title, fill=(0, 0, 0), font=font)
    
    if save is not None:
        pil_img.save(save)
        return None
    elif ret_pic:
        return pil_img
    else:
        return None


def plot_depth_masks_fast(
    gt_depth, gt_mask, 
    pred_depth, pred_mask, 
    erode=5,
    save=None,
    ret_pic=False,
):
    """Fast OpenCV/PIL version, roughly 10-50x faster than matplotlib."""
    gt_mask_e = erode_mask(gt_mask, kernel_size=erode)
    pred_mask_e = erode_mask(pred_mask, kernel_size=erode)
    gt_depth = to_numpy(gt_depth)
    gt_mask = to_numpy(gt_mask).astype(bool)
    pred_depth = to_numpy(pred_depth)
    pred_mask = to_numpy(pred_mask).astype(bool)
    gt_mask_e = to_numpy(gt_mask_e).astype(bool)
    pred_mask_e = to_numpy(pred_mask_e).astype(bool)
    
    if gt_mask_e.sum() == 0:
        gt_dmax = gt_dmin = 0
    else:
        gt_dmax = np.max(gt_depth[gt_mask_e])
        gt_dmin = np.min(gt_depth[gt_mask_e])
    if pred_mask_e.sum() == 0:
        pred_dmax = pred_dmin = 0
    else:
        pred_dmax = np.max(pred_depth[pred_mask_e])
        pred_dmin = np.min(pred_depth[pred_mask_e])
    
    # Convert depth maps to RGB visualization
    gt_depth_masked = np.where(gt_mask_e, gt_depth, 0)
    pred_depth_masked = np.where(pred_mask_e, pred_depth, 0)
    gt_depth_rgb = (colorize_depth(gt_depth_masked, valid_mask=gt_mask_e).clip(0, 1) * 255).astype(np.uint8)
    pred_depth_rgb = (colorize_depth(pred_depth_masked, valid_mask=pred_mask_e).clip(0, 1) * 255).astype(np.uint8)
    
    # Convert masks to uint8
    gt_mask_u8 = (gt_mask.astype(np.uint8) * 255)
    pred_mask_u8 = (pred_mask.astype(np.uint8) * 255)
    gt_mask_rgb = cv2.cvtColor(gt_mask_u8, cv2.COLOR_GRAY2RGB)
    pred_mask_rgb = cv2.cvtColor(pred_mask_u8, cv2.COLOR_GRAY2RGB)
    
    # Depth overlapped
    depth_overlap = cv2.addWeighted(gt_depth_rgb, 0.7, pred_depth_rgb, 0.5, 0)
    
    # Mask overlapped
    mask_overlap = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    mask_overlap[..., 0] = gt_mask_u8    # Red for GT
    mask_overlap[..., 1] = pred_mask_u8  # Green for pred
    
    # Create grid layout
    h, w = gt_depth_rgb.shape[:2]
    grid = np.ones((h * 3, w * 2, 3), dtype=np.uint8) * 255
    
    # Place images
    grid[0:h, 0:w] = pred_depth_rgb
    grid[0:h, w:w*2] = pred_mask_rgb
    grid[h:h*2, 0:w] = gt_depth_rgb
    grid[h:h*2, w:w*2] = gt_mask_rgb
    grid[h*2:h*3, 0:w] = depth_overlap
    grid[h*2:h*3, w:w*2] = mask_overlap
    
    # Add titles using PIL
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    titles = [
        (10, 5, f'Predicted Depth [m={pred_dmin:.3f} M={pred_dmax:.3f}]'),
        (w + 10, 5, 'Predicted Mask'),
        (10, h + 5, f'GT Depth [m={gt_dmin:.3f} M={gt_dmax:.3f}]'),
        (w + 10, h + 5, 'Ground Truth Mask'),
        (10, h*2 + 5, 'Depth Overlapped'),
        (w + 10, h*2 + 5, 'Mask Overlapped'),
    ]
    
    for x, y, title in titles:
        draw.text((x, y), title, fill=(0, 0, 0), font=font)
    
    if save is not None:
        pil_img.save(save)
        return None
    elif ret_pic:
        return pil_img
    else:
        return None
