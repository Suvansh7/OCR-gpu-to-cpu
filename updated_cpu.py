import cv2
import os
import time
import keras_ocr
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line ensures TensorFlow ignores any GPU.

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if not physical_devices:
    print("Running on CPU.")
else:
    print("Error: GPU is still visible.")

pipeline = keras_ocr.pipeline.Pipeline()

converter = tf.lite.TFLiteConverter.from_keras_model(pipeline.detector.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimization
tflite_model = converter.convert()

def preprocess_image(image, target_width=600):
    h, w, _ = image.shape
    scale_factor = target_width / w
    image_resized = cv2.resize(image, (target_width, int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    return image_resized

def process_frame_batch(frames_batch, pipeline, out, output_dims):
    images_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_batch]
    predictions_batch = pipeline.recognize(images_rgb)

    for idx, predictions in enumerate(predictions_batch):
        for text, box in predictions:
            box = box.astype(int)
            cv2.polylines(frames_batch[idx], [box], True, (0, 255, 0), 2)
            cv2.putText(frames_batch[idx], text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_resized_for_output = cv2.resize(frames_batch[idx], output_dims, interpolation=cv2.INTER_LINEAR)
        
        out.write(frame_resized_for_output)

# Optimized function with frame skipping and multithreading
def process_video_with_cpu(input_video, output_video, batch_size=8, frame_skip_interval=4):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width = 600
    scale_factor = target_width / frame_width
    processed_frame_height = int(frame_height * scale_factor)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, processed_frame_height))
    
    frame_count = 0
    total_time = 0
    frames_batch = []
    current_frame_index = 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_index += 1
            
            if current_frame_index % frame_skip_interval != 0:
                continue

            preprocessed_frame = preprocess_image(frame, target_width=target_width)
            frames_batch.append(preprocessed_frame)
            
            if len(frames_batch) == batch_size:
                start_time = time.time()
                
                executor.submit(process_frame_batch, frames_batch, pipeline, out, (target_width, processed_frame_height))

                end_time = time.time()
                total_time += end_time - start_time
                frame_count += len(frames_batch)
                frames_batch = []  

    cap.release()
    out.release()
    
    avg_fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds (CPU). Average FPS: {avg_fps:.2f}")

input_video = 'input.mp4'
output_video_cpu = 'output_optimized_cpu.avi'

process_video_with_cpu(input_video, output_video_cpu, batch_size=8, frame_skip_interval=4)
