import os
import shutil
import numpy as np
import cv2

def video_to_frames(video, input_path, output_path):
    temp_dir = os.path.join(output_path, 'temporary_images')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    video_path = os.path.join(input_path, video)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 3 == 0:  
            extracted_frames.append(frame)

    cap.release()

    if len(extracted_frames) < 240:
        last_frame = extracted_frames[-1] if extracted_frames else np.zeros((224, 224, 3), dtype=np.uint8)
        padding_needed = 240 - len(extracted_frames)
        extracted_frames.extend([last_frame] * padding_needed)

    extracted_frames = extracted_frames[:240]
    image_paths = []
    for idx, frame in enumerate(extracted_frames):
        frame_path = os.path.join(temp_dir, f'frame{idx}.jpg')
        cv2.imwrite(frame_path, frame)
        image_paths.append(frame_path)

    print(f'Total frames generated: {len(image_paths)}')
    return image_paths

