import cv2
import os
import time
import keras_ocr
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Set environment variable to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line ensures TensorFlow ignores any GPU.

# Check to verify that only the CPU is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if not physical_devices:
    print("Running on CPU.")
else:
    print("Error: GPU is still visible.")

# Load the Keras-OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Quantization (Post-Training)
converter = tf.lite.TFLiteConverter.from_keras_model(pipeline.detector.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimization
tflite_model = converter.convert()

# Preprocessing function to resize input images
def preprocess_image(image, target_width=600):
    # Resize image to a smaller width to reduce processing time
    h, w, _ = image.shape
    scale_factor = target_width / w
    image_resized = cv2.resize(image, (target_width, int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    return image_resized

# Threaded function to handle video reading and processing
def process_frame_batch(frames_batch, pipeline, out, output_dims):
    images_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_batch]
    predictions_batch = pipeline.recognize(images_rgb)

    for idx, predictions in enumerate(predictions_batch):
        for text, box in predictions:
            box = box.astype(int)
            cv2.polylines(frames_batch[idx], [box], True, (0, 255, 0), 2)
            cv2.putText(frames_batch[idx], text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Resize processed frame to match output dimensions before writing
        frame_resized_for_output = cv2.resize(frames_batch[idx], output_dims, interpolation=cv2.INTER_LINEAR)
        # Write processed frame to the output video
        out.write(frame_resized_for_output)

# Optimized function with frame skipping and multithreading
def process_video_with_cpu(input_video, output_video, batch_size=8, frame_skip_interval=4):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get original frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize dimensions for processing
    target_width = 600
    scale_factor = target_width / frame_width
    processed_frame_height = int(frame_height * scale_factor)
    
    # Initialize VideoWriter with processed frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, processed_frame_height))
    
    frame_count = 0
    total_time = 0
    frames_batch = []
    current_frame_index = 0
    
    # ThreadPoolExecutor for concurrent frame processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_index += 1
            # Skip frames based on the frame_skip_interval
            if current_frame_index % frame_skip_interval != 0:
                continue

            # Preprocess the frame by resizing
            preprocessed_frame = preprocess_image(frame, target_width=target_width)
            frames_batch.append(preprocessed_frame)
            
            if len(frames_batch) == batch_size:
                start_time = time.time()
                
                # Submit frame batch for parallel processing
                executor.submit(process_frame_batch, frames_batch, pipeline, out, (target_width, processed_frame_height))

                end_time = time.time()
                total_time += end_time - start_time
                frame_count += len(frames_batch)
                frames_batch = []  # Reset the batch

    cap.release()
    out.release()
    
    avg_fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds (CPU). Average FPS: {avg_fps:.2f}")

# Provide input and output video paths
input_video = 'input.mp4'
output_video_cpu = 'output_optimized_cpu.avi'

# Adjust the frame_skip_interval and batch_size for better performance
process_video_with_cpu(input_video, output_video_cpu, batch_size=8, frame_skip_interval=4)
