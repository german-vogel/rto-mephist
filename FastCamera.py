import os
import cv2
import numpy as np

def create_slow_motion_video(input_folder, output_file, original_fps=10000, playback_fps=10, font_scale=0.8):
    """
    Create slow-motion video from fast camera images while maintaining accurate timestamps.
    
    Args:
        input_folder: Path to folder containing JPG images
        output_file: Output video path (.avi)
        original_fps: Original camera frame rate (default 10000 fps)
        playback_fps: Desired output frame rate (default 30 fps for smooth slow-motion)
        font_scale: Text size scaling factor
    """
    # Get sorted image files
    image_files = sorted([f for f in os.listdir(input_folder) 
                        if f.lower().endswith('.jpg')],
                        key=lambda x: int(x.split('S0001')[-1].split('.')[0]))
    
    if not image_files:
        print(f"No JPG images found in {input_folder}")
        return

    # Load first image to get dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    if first_image is None:
        print(f"Error reading first image {image_files[0]}")
        return
    
    height, width = first_image.shape[:2]
    discharge_num = image_files[0].split('M_')[0]

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

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)  # White text
    thickness = 2

    print(f"Processing {len(image_files)} frames...")
    
    try:
        for frame_num, image_file in enumerate(image_files):
            img = cv2.imread(os.path.join(input_folder, image_file))
            if img is None:
                print(f"Warning: Could not read {image_file}, skipping")
                continue
                
            # Calculate actual time in milliseconds
            actual_time_ms = frame_num * (1000 / original_fps)  # More precise calculation
            
            # Add timestamp and discharge info (bottom left)
            y_pos = height - 30
            cv2.putText(img, f"Discharge: {discharge_num}", (20, y_pos), 
                       font, font_scale, font_color, thickness, cv2.LINE_AA)
            cv2.putText(img, f"Time: {actual_time_ms:.2f} ms", (20, y_pos - 50), 
                       font, font_scale, font_color, thickness, cv2.LINE_AA)
            
            # Write frame to video
            video_writer.write(img)
            
            # Progress reporting
            if (frame_num + 1) % 100 == 0 or (frame_num + 1) == len(image_files):
                print(f"Processed {frame_num + 1}/{len(image_files)} frames "
                      f"({actual_time_ms:.2f} ms)")
    
    finally:
        video_writer.release()
        print(f"Slow-motion video saved to {output_file}")
        print(f"Playback speed: {slow_motion_factor:.1f}x slower than real-time")

if __name__ == "__main__":
    discharge_number = "2623"  # Change this for different discharges
    input_folder = f"C:\\Users\\vogel\\Downloads\\{discharge_number}_FastCamera"
    output_file = f"C:\\Users\\vogel\\Downloads\\{discharge_number}_10fps.avi"
    
    # Adjust these parameters as needed:
    original_camera_fps = 10000  # Your camera's actual frame rate
    playback_fps = 10
    font_scale = 0.8
    
    create_slow_motion_video(
        input_folder, 
        output_file,
        original_fps=original_camera_fps,
        playback_fps=playback_fps
    )