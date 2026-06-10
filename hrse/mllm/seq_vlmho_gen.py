import os
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def colorize_depth(depth_map, ctab='turbo'):
    """
    Colorizes a depth map using a colormap.
    
    Args:
        depth_map (numpy.ndarray): The depth map to colorize.
        ctab (str): The colormap to use. Default is 'turbo'.
        
    Returns:
        numpy.ndarray: The colorized depth map.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Normalize the depth map to [0, 1]
    norm_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    
    # Apply the colormap
    cmap = cm.get_cmap(ctab)
    colorized_depth = cmap(norm_depth)[:, :, :3]  # Get RGB channels
    
    return (colorized_depth * 255).astype(np.uint8)

def read_images_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    img_extensions = {'.jpg', '.jpeg', '.png',}
    img_files = [f for f in files if os.path.splitext(f.lower())[1] in img_extensions]
    sorted_img_files = natsorted(img_files)
    return [os.path.join(folder_path, f) for f in sorted_img_files]

def read_depths_from_file(file_path):
    depths = np.load(file_path, allow_pickle=True)
    return depths

def annotate_image(img : Image.Image, img_path: str, text_position_mode: str = "top-left"):
    # img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    filename = os.path.basename(img_path)
    
    font_size = int(min(img.width, img.height) * 0.07)  # Font size is 7% of the image size.
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # Compute the annotation position, with four corner options.
    margin_x = int(img.width * 0.05)
    margin_y = int(img.height * 0.05)
    bbox = draw.textbbox((0, 0), filename, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    if text_position_mode == "top-right":
        text_position = (img.width - margin_x - text_width, margin_y)
    elif text_position_mode == "bottom-left":
        text_position = (margin_x, img.height - margin_y - text_height)
    elif text_position_mode == "bottom-right":
        text_position = (img.width - margin_x - text_width, img.height - margin_y - text_height)
    else:
        text_position = (margin_x, margin_y)
    
    # Add text in white with a black outline for readability.
    # Add a black shadow.
    shadow_offset = max(1, int(font_size * 0.05))
    draw.text((text_position[0]+shadow_offset, text_position[1]+shadow_offset), filename, font=font, fill="black")
    # Add the main text.
    draw.text(text_position, filename, font=font, fill="white")
    
    return img

def merge_images_horizontally(image_tuples, text_position_mode: str = "top-left"):
    ret = []
    for tup in image_tuples:
        N = len(tup)
        # Add annotations and collect all images.
        annotated_images = [annotate_image(img, path, text_position_mode=text_position_mode) for img, path in tup]
    
        # Find the maximum height for normalization.
        max_height = max(img.height for img in annotated_images)
        
        # Resize all images to the maximum height while preserving aspect ratio.
        resized_images = []
        for img in annotated_images:
            ratio = max_height / img.height
            new_width = int(img.width * ratio)
            resized_images.append(img.resize((new_width, max_height), Image.LANCZOS))
    
        # Width of the black separator.
        separator_width = 12
        
        # Compute the total merged image width, including separators.
        total_width = sum(img.width for img in resized_images)
        if N > 1:  # Add separators only when there are multiple images.
            total_width += (N - 1) * separator_width
    
        # Create a new image.
        merged_image = Image.new('RGB', (total_width, max_height))
        
        # Paste resized images into the new image and add black separators.
        x_offset = 0
        for i, img in enumerate(resized_images):
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
            
            # Add a black separator after each image except the last one.
            if i < len(resized_images) - 1 and N > 1:
                # Keep the separator area black, which is the default background.
                x_offset += separator_width
        ret.append(merged_image)
    return ret

def merge_images_vertically(imgs):
    if not imgs:
        return None

    N = len(imgs)
    # Height of the black separator.
    separator_height = 12
    
    # Compute the total merged image height, including separators.
    total_height = sum(img.height for img in imgs)
    if N > 1:
        total_height += (N - 1) * separator_height
        
    max_width = max(img.width for img in imgs)
    
    # Create a new image.
    merged_image = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for i, img in enumerate(imgs):
        # Paste the image centered horizontally.
        x_pos = (max_width - img.width) // 2
        merged_image.paste(img, (x_pos, y_offset))
        y_offset += img.height
        
        # Add a black separator below each image except the last one.
        if i < N - 1:
            y_offset += separator_height
    
    return merged_image

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge and annotate images")
    parser.add_argument("-i", "--seq-path", help="Sequence folder path")
    parser.add_argument("-o", "--output", default="output/mllm/", help="Output path, defaults to output/mllm/")
    parser.add_argument("-k", "--count", type=int, default=3, help="Number of images to merge, defaults to 3")
    parser.add_argument("--interval", type=int, default=1, help="Interval between images in frames, defaults to 1")
    parser.add_argument("--rand-k", action="store_true", help="Randomly select k images to merge")
    parser.add_argument(
        "--text-position",
        type=str,
        default="top-left",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        help="Annotation text position: top-left/top-right/bottom-left/bottom-right, defaults to top-left",
    )
    
    args = parser.parse_args()
    
    img_paths = read_images_from_folder(f"{args.seq_path}/build/image")
    seq_name = os.path.basename(args.seq_path.strip('/'))
    os.makedirs(f'{args.output}/{seq_name}', exist_ok=True)

    K = args.count
    interval = int(args.interval)
    # Automatically compute the maximum number of compositing rounds.
    max_round = max(1, (len(img_paths) - (K - 1) * interval))
    # Sequentially sample the start frames.
    start_idx = 0
    samples = list(range(start_idx, max_round))
    
    depths = read_depths_from_file(os.path.join(args.seq_path, "build/metric_depth.pkl"))
    rgbs = [Image.open(path) for path in img_paths]
    depths_colorized = [colorize_depth(depths[i]) for i in range(len(depths))]
    
    img_tups = []
    dep_tups = []
    
    max_start = len(img_paths) - (K - 1) * interval
    for start in range(max_start):
        selected_indices = [start + interval * i for i in range(K)]
        img_tups.append([(rgbs[i], img_paths[i]) for i in selected_indices])
        dep_tups.append([(Image.fromarray(depths_colorized[i]), img_paths[i]) for i in selected_indices])
    
    merged_rgb = merge_images_horizontally(img_tups, text_position_mode=args.text_position)
    merged_depth = merge_images_horizontally(dep_tups, text_position_mode=args.text_position)
    
    for i, selected in enumerate(merged_rgb):
        output_path = f"{args.output}/{seq_name}/sampleRGB_{i+1}.jpg"
        selected.save(output_path)
        composite = merge_images_vertically([selected, merged_depth[i]])
        composite_output_path = f"{args.output}/{seq_name}/sampleRGBD_{i+1}.jpg"
        composite.save(composite_output_path)

if __name__ == "__main__":
    main()


"""
python seq_vlmho_gen.py \
    -i ds/rsrd/rs_candybox2 \
    -o output/mllm/ \
    -k 3 --interval 1
"""
