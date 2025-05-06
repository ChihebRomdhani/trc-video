import os
import cv2
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from moviepy.video.fx import *  # Updated import for crop effect

# --- Augmentation Functions ---

def flip_video(input_path, output_path, flip_code):
  
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, flip_code)
        out.write(flipped)

    cap.release()
    out.release()
    print(f"Saved flipped video: {output_path}")

def adjust_colors(input_path, output_path, brightness=0, red_shift=0):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to float for safe arithmetic
        frame = frame.astype(np.float32)
        # Adjust brightness
        frame += brightness
        # Add red shift: increase the red channel (channel index 2 in BGR)
        frame[:, :, 2] += red_shift
        # Clip values to valid range [0, 255] and convert back to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved color adjusted video: {output_path}")

def zoom_effect(clip, zoom_factor):

    w, h = clip.size
    # Calculate new dimensions for the crop
    new_w = w / zoom_factor
    new_h = h / zoom_factor
    x_center = w / 2
    y_center = h / 2
    x1 = x_center - new_w / 2
    y1 = y_center - new_h / 2
    x2 = x1 + new_w
    y2 = y1 + new_h
    # Crop the clip and then resize to original dimensions
    return clip.fx(all.crop, x1=x1, y1=y1, x2=x2, y2=y2).resize((w, h))

def zoom_video(input_path, output_path, zoom_factor=1.1):
 
    clip = VideoFileClip(input_path)
    zoomed = zoom_effect(clip, zoom_factor)
    zoomed.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    clip.close()
    zoomed.close()
    print(f"Saved zoomed video: {output_path}")

def add_gaussian_noise(input_path, output_path, mean=0, std=10):
 
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        noise = np.random.normal(mean, std, frame.shape).astype(np.float32)
        noisy_frame = frame.astype(np.float32) + noise
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        out.write(noisy_frame)

    cap.release()
    out.release()
    print(f"Saved gaussian noise video: {output_path}")

def adjust_contrast_saturation(input_path, output_path, contrast=1.2, saturation=1.2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Adjust saturation: multiply the S channel by the saturation factor
        hsv[:, :, 1] *= saturation
        # Adjust contrast by modifying the V channel
        hsv[:, :, 2] *= contrast
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        frame_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(frame_adjusted)

    cap.release()
    out.release()
    print(f"Saved contrast and saturation jitter video: {output_path}")


def augment_video_row(row, video_col, aug_dir):

    # Ensure the video path is treated as a string
    orig_path = str(row[video_col]).strip()
    if not os.path.isfile(orig_path):
        print(f"Video file not found: {orig_path}")
        return []
    
    base, ext = os.path.splitext(orig_path)
    new_rows = []
    
    # List of augmentations to perform.
    augmentations = [
        {"name": "flip_h", "func": flip_video, "params": {"flip_code": 1}},
        {"name": "flip_v", "func": flip_video, "params": {"flip_code": 0}},
        {"name": "flip_both", "func": flip_video, "params": {"flip_code": -1}},
        {"name": "darker", "func": adjust_colors, "params": {"brightness": -30, "red_shift": 0}},
        {"name": "red_tint", "func": adjust_colors, "params": {"brightness": 0, "red_shift": 40}},
        {"name": "brighter", "func": adjust_colors, "params": {"brightness": 30, "red_shift": 0}},
        {"name": "zoom", "func": zoom_video, "params": {"zoom_factor": 1.1}},
        {"name": "gaussian_noise", "func": add_gaussian_noise, "params": {"mean": 0, "std": 10}},
        {"name": "contrast_saturation", "func": adjust_contrast_saturation, "params": {"contrast": 1.2, "saturation": 1.2}},
    ]
    
    for aug in augmentations:
        new_filename = f"{base}_{aug['name']}{ext}"
        new_filepath = os.path.join(aug_dir, os.path.basename(new_filename))
        os.makedirs(aug_dir, exist_ok=True)
        
        try:
            aug["func"](orig_path, new_filepath, **aug["params"])
        except Exception as e:
            print(f"Error augmenting {orig_path} with {aug['name']}: {e}")
            continue
        
        new_row = row.copy()
        new_row[video_col] = new_filepath
        new_rows.append(new_row)
    
    return new_rows

def augment_videos_in_csv(csv_path, video_columns, aug_dir, output_csv):

    df = pd.read_csv(csv_path, delimiter=',')
    augmented_rows = []
    
    for idx, row in df.iterrows():
        for vcol in video_columns:
            new_rows = augment_video_row(row, vcol, aug_dir)
            augmented_rows.extend(new_rows)
    
    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows)
        df_combined = pd.concat([df, df_aug], ignore_index=True)
    else:
        df_combined = df.copy()
    
    df_combined.to_csv(output_csv, index=False)
    print(f"Augmented CSV saved to {output_csv}")

# --- Example Usage ---

if __name__ == '__main__':
    csv_path = "processed_data_redownloaded.csv"
    # List your video column names here (for example, if there are four columns)
    video_columns = [
        "pulpe de l´index (main droite)  ",
        "pulpe de l´index (main gauche) ",
        "l´éminence thénar  à droite ",
        "l´éminence thénar  à gauche "
    ]
    
    aug_dir = "downloaded_videos"
    output_csv = "processed_data_augmented2.csv"
    
    augment_videos_in_csv(csv_path, video_columns, aug_dir, output_csv)
