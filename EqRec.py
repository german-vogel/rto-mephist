import os
import cv2
import numpy as np

def create_slow_motion_video(input_folder, output_file, original_fps=10000, playback_fps=10):
    """
    Create slow-motion video from fast camera images without any text overlays.
    
    Args:
        input_folder: Path to folder containing JPG images
        output_file: Output video path (.avi)
        original_fps: Original camera frame rate (default 10000 fps)
        playback_fps: Desired output frame rate (default 10 fps for smooth slow-motion)
    """
    # Get sorted image files (1.JPG, 2.JPG, 3.JPG, etc.)
    image_files = sorted([f for f in os.listdir(input_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                        key=lambda x: int(os.path.splitext(x)[0]))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    # Load first image to get dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    if first_image is None:
        print(f"Error reading first image {image_files[0]}")
        return
    
    height, width = first_image.shape[:2]
    print(f"Detected image size: {width}x{height}")

    # Calculate slow-motion factor
    slow_motion_factor = original_fps / playback_fps
    print(f"Creating slow-motion video ({slow_motion_factor:.1f}x slower)")
    print(f"Original: {original_fps} fps | Playback: {playback_fps} fps")

    # Video writer setup (MJPG codec works best for AVI)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_file, fourcc, playback_fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not initialize video writer")
        return

    print(f"Processing {len(image_files)} frames...")
    
    try:
        for frame_num, image_file in enumerate(image_files):
            img = cv2.imread(os.path.join(input_folder, image_file))
            if img is None:
                print(f"Warning: Could not read {image_file}, skipping")
                continue
            
            # Write frame to video (no text added)
            video_writer.write(img)
            
            # Progress reporting
            if (frame_num + 1) % 10 == 0 or (frame_num + 1) == len(image_files):
                actual_time_ms = frame_num * (1000 / original_fps)
                print(f"Processed {frame_num + 1}/{len(image_files)} frames "
                      f"({actual_time_ms:.2f} ms)")
    
    finally:
        video_writer.release()
        print(f"Slow-motion video saved to {output_file}")
        print(f"Playback speed: {slow_motion_factor:.1f}x slower than real-time")

if __name__ == "__main__":
    discharge_number = "2628"  # Change this for different discharges
    input_folder = f"C:\\Users\\vogel\\Downloads\\{discharge_number}_EquiRecons"
    output_file = f"C:\\Users\\vogel\\Downloads\\{discharge_number}_10fps.avi"
    
    # Adjust these parameters as needed:
    original_camera_fps = 10000  # Your camera's actual frame rate
    playback_fps = 10
    
    create_slow_motion_video(
        input_folder, 
        output_file,
        original_fps=original_camera_fps,
        playback_fps=playback_fps
    )