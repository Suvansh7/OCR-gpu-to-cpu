import cv2
import time
import keras_ocr
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    print("GPU Available: Yes")
else:
    print("GPU Available: No")

pipeline = keras_ocr.pipeline.Pipeline()

def process_video_with_gpu(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    total_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = pipeline.recognize([image])[0]
        
        for text, box in predictions:
            box = box.astype(int)
            cv2.polylines(frame, [box], True, (0, 255, 0), 2)
            cv2.putText(frame, text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        end_time = time.time()
        total_time += end_time - start_time
        frame_count += 1
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    avg_fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds (GPU). Average FPS: {avg_fps:.2f}")

input_video = 'input.mp4'
output_video_gpu = 'output_gpu.avi'

process_video_with_gpu(input_video, output_video_gpu)
