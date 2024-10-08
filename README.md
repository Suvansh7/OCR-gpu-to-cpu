
# **Optimizing Keras-OCR for CPU-Based Video Processing**

This repository demonstrates how to optimize the [Keras-OCR](https://github.com/faustomorales/keras-ocr) pipeline for processing video frames using a CPU. The goal is to reduce processing time, increase average frames per second (FPS), and handle large videos efficiently without compromising recognition accuracy.

## **Features and Optimizations**

### 1. **Disabling GPU for CPU Execution**
   - We force TensorFlow to ignore any GPU and run the Keras-OCR pipeline solely on the CPU by setting the environment variable `CUDA_VISIBLE_DEVICES = "-1"`. This ensures the model runs in CPU-only environments.

### 2. **Model Quantization (Post-Training)**
   - Model quantization reduces the size and computational load of the model. We convert the Keras-OCR detection model to TensorFlow Lite using post-training quantization (`TFLiteConverter`). This optimization improves inference time on the CPU without significantly affecting the model’s accuracy.

### 3. **Image Preprocessing and Resizing**
   - To reduce the computational burden, input video frames are resized to a target width (600px) before processing. This preprocessing step lowers the amount of data the model has to handle, leading to faster execution times.

### 4. **Frame Skipping**
   - A frame skipping mechanism is applied to process every 4th frame instead of every single frame. This reduces the overall processing time by skipping redundant frames, making the process more efficient without major loss of information in text recognition.

### 5. **Batch Processing**
   - Frames are processed in batches, which speeds up execution by grouping multiple frames and performing text recognition on them in parallel. This improves CPU usage efficiency compared to processing each frame individually.

### 6. **Multithreading with ThreadPoolExecutor**
   - To further improve performance, multithreading is employed using Python's `ThreadPoolExecutor`. Multiple batches of frames are processed concurrently, which maximizes CPU utilization and reduces idle time between batches.

### 7. **Optimized Output Handling**
   - After processing, frames are resized to match the output video dimensions, ensuring consistency between input and output. Processed frames are written to the output video file efficiently.

### 8. **Accurate Time and FPS Measurement**
   - The total processing time and the average FPS are computed to benchmark the performance of the optimized pipeline. This provides insight into how well the optimizations have improved the efficiency of the system.

## **How to Use**

1. Install the necessary dependencies:
   ```bash
   pip install keras-ocr opencv-python-headless tensorflow
   ```

2. Run the `process_video_with_cpu` function to process a video with frame skipping and batch processing:
   ```python
   input_video = 'input.mp4'
   output_video_cpu = 'output_optimized_cpu.avi'
   process_video_with_cpu(input_video, output_video_cpu, batch_size=8, frame_skip_interval=4)
   ```

3. The output video will be saved, and the terminal will display the average FPS and total processing time.

## **Performance Impact**
With these optimizations, we observe:
   - **Reduced processing time** for CPU-only environments.
   - **Higher average FPS** due to frame skipping and efficient multithreading.
   - **Lower computational load** by resizing input images and using model quantization.
   
These techniques collectively improve the performance of the Keras-OCR pipeline when running on a CPU, making it suitable for environments where GPU resources are limited or unavailable.
