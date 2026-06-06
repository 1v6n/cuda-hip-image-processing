# CUDA/HIP Image Processing

This project implements image filtering algorithms in both CPU and GPU versions to compare their performance. The current pipeline supports converting images to grayscale, applying a Gaussian filter, and running a Sobel edge detection filter.

## Project Structure

* `hw3_ex1.cu`: Main source file containing the CPU reference functions, GPU kernels, and execution harness.
* `data/`: Directory containing input BMP images (e.g. rome.bmp, hk.bmp, nyc.bmp). This folder is ignored by Git.
* `images/`: Directory where output images are written. This folder is ignored by Git.

## Compilation

The codebase can be compiled using either NVIDIA's NVCC or AMD's HIPCC compiler depending on the hardware platform.

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

### Outputs

The application outputs processing times for both CPU and GPU steps, and saves the resulting images to the `images/` directory:

* `images/hw3_result_cpu_1.bmp`, `images/hw3_result_cpu_2.bmp`, and `images/hw3_result_cpu_3.bmp` are saved for the CPU implementation of all three steps (Grayscale, Gaussian, Sobel).
* `images/hw3_result_gpu_1.bmp` is saved for the GPU grayscale implementation. (GPU outputs for step 2 and 3 will be saved once their corresponding kernels are implemented and enabled in main).
