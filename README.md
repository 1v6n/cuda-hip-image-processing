# CUDA/HIP Image Processing

This project implements image filtering algorithms in both CPU and GPU versions to compare their performance. The pipeline converts an input image to grayscale, applies a Gaussian blur, and runs a Sobel edge detection filter.

## Project Structure

* `hw3_ex1.cu`: Main source file containing CPU reference functions, GPU kernels (Global and Shared Memory versions), and the execution harness.
* `data/`: Directory containing input BMP images (ignored by Git).
* `images/`: Directory where output images are written (ignored by Git).

## Compilation

The codebase is dual-compatible and can be compiled using either AMD's HIPCC or NVIDIA's NVCC compiler.

### AMD HIPCC
To compile natively for AMD GPUs:
```bash
hipcc hw3_ex1.cu -o hw3_ex1
```

### NVIDIA NVCC
To compile natively for NVIDIA GPUs:
```bash
nvcc hw3_ex1.cu -o hw3_ex1 -lm
```

## Running the Application

Run the compiled executable by passing the path to the input BMP file:

```bash
./hw3_ex1 data/rome.bmp
```

---

## Performance Benchmark Logs

Below are the execution times measured on an AMD Ryzen APU (gfx1103 target, Radeon 780M GPU) using a 3300x2468 input BMP file (approx. 8.14 million pixels).

### Step-by-Step Benchmarks
* **Step 1: Grayscale Conversion**
  * CPU: 9.068 ms
  * GPU (with PCIe copy): 9.350 ms
* **Step 2: Gaussian Blur (3x3)**
  * CPU: 8.195 ms
  * GPU Global (with PCIe copy): 8.204 ms
  * GPU Shared Memory (with PCIe copy): 8.893 ms
* **Step 3: Sobel Edge Detection (3x3)**
  * CPU: 8.827 ms
  * GPU Global (with PCIe copy): 4.628 ms
  * GPU Shared Memory (with PCIe copy): 13.290 ms

### Pipeline Summary Table
The table below compares the total continuous pipeline execution (Warm Caches CPU vs. Pipelined GPU executing all stages on the device back-to-back with a single final memory copy):

| Execution Type | Time (ms) | Speedup |
| :--- | :--- | :--- |
| CPU Pipeline (Warm Caches, 1 block) | 23.507 ms | Reference (1.00x) |
| GPU Pipeline (Back-to-back, 1 copy) | 12.356 ms | 1.90x |
