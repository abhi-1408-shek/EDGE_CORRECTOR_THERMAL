# Detailed Project Report: The "Software Edge" Real-Time Hybrid Image Enhancement Pipeline

## 1. Executive Summary
This project aims to develop a high-performance, software-only implementation of a Hybrid Deterministic and AI-Based Edge Correction pipeline. Originally conceptualized for FPGA hardware, this project translates the architecture to run in real-time on resource-constrained Edge GPUs (e.g., NVIDIA Jetson series) or standard x86 CPUs. The system cleans noisy sensor data (such as thermal or low-light imagery) by combining the stability of traditional physics-based filters with the detail-recovery power of Deep Learning, all while maintaining strict real-time performance (30–60 FPS) and preventing AI hallucinations.

## 2. Problem Statement
Raw imaging sensors in low-light or thermal conditions produce highly noisy data. Traditional deterministic filters (like Gaussian or Wiener) reduce noise but heavily blur the edges of objects, making downstream detection difficult. Conversely, purely AI-based enhancement can reconstruct fine details but is computationally expensive and prone to "hallucinating" textures that don't exist in reality. 
**The Solution:** A hybrid pipeline that applies AI enhancement *only* where structural edges exist, relying on fast, predictable deterministic filters for the rest of the image. The software challenge is executing both pipelines concurrently without stalling the frame rate.

---

## 3. System Architecture
The pipeline consists of five interconnected stages, partitioned across the CPU and GPU to maximize throughput.

### Stage 1: Deterministic Pre-Processing (CPU via OpenCV)
*   **Function:** Applies traditional algorithms to suppress baseline noise and correct pixel errors.
*   **Components:**
    *   **Dead Pixel Correction:** Median filtering to remove sensor anomalies.
    *   **Wiener/Bilateral Filtering:** Suppresses high-frequency noise while attempting to preserve major intensity shifts.
*   **Execution:** Highly optimized C++ OpenCV routines utilizing SIMD instructions (AVX2/NEON) on the CPU.

### Stage 2: Edge Map Extraction (CPU via OpenCV)
*   **Function:** Generates an "Edge Confidence Map" to guide the AI and Fusion stages.
*   **Components:** Sobel or Canny edge detectors applied to the Stage 1 output. This creates a grayscale mask where white pixels represent strong edges and black pixels represent flat/uniform areas.

### Stage 3: AI-Based Correction (GPU via TensorRT)
*   **Function:** Recovers fine details and sharpens boundaries that were smoothed out by the sensor or Stage 1 filters.
*   **Components:** A lightweight Convolutional Neural Network (CNN), inspired by MobileNet or a miniature U-Net. 
*   **Execution:** The model is highly compressed (INT8 Quantization) and executed using NVIDIA TensorRT for maximum inference speed on the GPU.

### Stage 4: Confidence-Weighted Fusion (GPU via CUDA)
*   **Function:** Blends the outputs of Stage 1 (Deterministic) and Stage 3 (AI) together.
*   **Mechanism:** Uses the Edge Map (Stage 2) as an alpha mask. 
    *   *High Edge Confidence (White mask):* 90% AI output, 10% Deterministic output.
    *   *Low Edge Confidence (Black mask):* 10% AI output, 90% Deterministic output.
*   **Execution:** A custom CUDA kernel performs this pixel-wise alpha blending near-instantly on the GPU.

### Stage 5: Validation (CPU)
*   **Function:** Anti-hallucination safeguard.
*   **Mechanism:** Compares the final fused image against the original raw sensor data. If the AI injected an edge that has absolutely no gradient representation in the raw data, the system flags it and reverts that specific bounding box to the deterministic output.

---

## 4. Hardware & Software Stack

### Hardware Target (The "Software Edge")
*   **Primary Target:** NVIDIA Jetson Nano / Jetson Orin Nano (representing a strict SWaP—Size, Weight, and Power—constrained edge device).
*   **Secondary Target:** Standard x86 PC with a consumer GPU (for development and benchmarking).

### Technology Stack
*   **Core Logic & Multithreading:** Modern C++ (C++17/20) for zero-overhead execution.
*   **Deterministic Filtering:** OpenCV (compiled with CUDA/OpenMP support).
*   **AI Training (Offline):** Python, PyTorch.
*   **AI Inference (Runtime):** NVIDIA TensorRT (for converting PyTorch `.onnx` models into highly optimized engines).
*   **Parallel Computing:** CUDA C++ (for writing custom fusion kernels).

---

## 5. Workload Partitioning & Concurrency Model
To achieve 60 FPS, the system cannot process images sequentially. It utilizes a **Producer-Consumer multithreading model**.

1.  **Thread 1 (Capture & Stage 1+2):** Grabs the frame from the camera, runs OpenCV Bilateral and Sobel filters on the CPU, and pushes the results to a thread-safe Queue.
2.  **Thread 2 (AI Inference):** Pulls from the Queue, copies data to the GPU (Host-to-Device), runs the TensorRT model, and stores the output in GPU memory.
3.  **Thread 3 (Fusion & Display):** Runs the CUDA fusion kernel on the GPU memory, copies the final result back to the CPU (Device-to-Host), and renders it to the screen.

*By pipelining these threads, the system's bottleneck is limited only to the slowest individual stage, rather than the sum of all stages.*

---

## 6. Model Optimization Strategy
A standard AI model cannot run at 60 FPS on edge hardware. The following compression techniques are mandatory:
1.  **Architecture:** Replacing standard Convolutions with Depthwise-Separable Convolutions.
2.  **Quantization-Aware Training (QAT):** Training the model in PyTorch while simulating INT8 (8-bit integer) precision. This allows the model to learn how to preserve edges without needing 32-bit floating-point math.
3.  **TensorRT Compilation:** Compiling the model specifically for the target GPU architecture, fusing layers (e.g., Conv + BatchNorm + ReLU merged into a single operation) to maximize memory bandwidth.

---

## 7. Phased Implementation Plan

*   **Phase 1: Python Prototyping (Weeks 1-3)**
    *   Build the entire 5-stage pipeline in Python using OpenCV and PyTorch.
    *   Establish a baseline for image quality (PSNR/SSIM metrics).
    *   *Deliverable:* A slow, but mathematically correct, proof-of-concept.
*   **Phase 2: AI Training & Export (Weeks 4-5)**
    *   Train the lightweight CNN on a dataset of noisy/clean images.
    *   Export the model to ONNX format and convert it to a TensorRT Engine.
*   **Phase 3: C++ Translation & OpenCV (Weeks 6-7)**
    *   Rewrite Stages 1, 2, and 5 in C++ using OpenCV. 
    *   Implement the multithreaded queue system.
*   **Phase 4: TensorRT & CUDA Integration (Weeks 8-9)**
    *   Integrate the TensorRT C++ API into the pipeline for Stage 3.
    *   Write the custom CUDA kernel for the Stage 4 Confidence Fusion.
*   **Phase 5: Profiling & Bottleneck Resolution (Weeks 10-12)**
    *   Use NVIDIA Nsight Systems to profile the CPU/GPU timeline.
    *   Resolve memory copy bottlenecks between Host and Device.
    *   *Deliverable:* Final, optimized real-time edge application.

---

## 8. Success Metrics
The project is considered successful if it meets the following criteria on the target edge device:
1.  **Throughput:** Maintains a steady frame rate of > 30 FPS for 640x480 resolution input.
2.  **Latency:** Glass-to-glass latency of less than 50 milliseconds.
3.  **Quality:** Outperforms standalone Bilateral filtering in edge-preservation metrics (Edge Preservation Index - EPI).
4.  **Reliability:** Stage 5 validation flags false edges with >95% accuracy.

## 9. Conclusion
The "Software Edge" pipeline provides a highly accessible, scalable alternative to custom FPGA development. By leveraging modern C++ concurrency, optimized OpenCV routines, and TensorRT AI acceleration, this project proves that advanced, hallucination-free hybrid image enhancement can be achieved in real-time on accessible, low-power computing platforms.
