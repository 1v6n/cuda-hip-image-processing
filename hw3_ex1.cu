
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define cudaMalloc hipMalloc
#define cudaMemset hipMemset
#define cudaMemcpy hipMemcpy
#define cudaFree hipFree
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaSuccess hipSuccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaError_t hipError_t
#endif

#define BLOCK_SIZE  16
#define HEADER_SIZE 138

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct
{
    int   width;
    int   height;
    float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(const char *filename)
{
    BMPImage bitmap = { 0 };
    int      size   = 0;
    BYTE     *data  = NULL;
    FILE     *file  = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24bpp)
    fread(g_info, sizeof(BYTE), HEADER_SIZE, file);

    // Get the image width / height from the header
    bitmap.width  = *((int *)&g_info[18]);
    bitmap.height = *((int *)&g_info[22]);
    size          = *((int *)&g_info[34]);
    
    // Read the image data
    data = (BYTE *)malloc(sizeof(BYTE) * size);
    fread(data, sizeof(BYTE), size, file);
    
    // Convert the pixel values to float
    bitmap.data = (float *)malloc(sizeof(float) * size);
    
    for (int i = 0; i < size; i++)
    {
        bitmap.data[i] = (float)data[i];
    }
    
    fclose(file);
    free(data);
    
    return bitmap;
}

/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int width, int height, float *image, const char *filename)
{
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < height; h++)
    {
        int offset = h * width;
        
        for (int w = 0; w < width; w++)
        {
            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                                               image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

/**
 * Releases a given BMPImage.
 */
void freeBMP(BMPImage bitmap)
{
    free(bitmap.data);
}

/**
 * Checks if there has been any CUDA error. The method will automatically print
 * some information and exit the program when an error is found.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define checkCUDAError() gpuErrchk(cudaGetLastError())

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/**
 * Stores the result image and prints a message.
 */
void store_result(int index, double elapsed_cpu, double elapsed_gpu,
                  int width, int height, float *image_cpu, float *image_gpu, 
                  int gpu_enabled, const char *gpu_suffix)
{
    char path_cpu[255];
    char path_gpu[255];
    
    if (elapsed_cpu > 0)
    {
        sprintf(path_cpu, "images/hw3_result_cpu_%d.bmp", index);
        writeBMPGrayscale(width, height, image_cpu, path_cpu);
        
        printf("Step #%d Completed:\n", index);
        printf("  CPU result stored in \"%s\" (Elapsed CPU: %fms)\n", path_cpu, elapsed_cpu);
    }
    
    if (!gpu_enabled)
    {
        printf("  [GPU version not available]\n");
    }
    else
    {
        sprintf(path_gpu, "images/hw3_result_gpu%s_%d.bmp", gpu_suffix, index);
        writeBMPGrayscale(width, height, image_gpu, path_gpu);
        printf("  GPU result stored in \"%s\" (Elapsed GPU: %fms)\n", path_gpu, elapsed_gpu);
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int width, int height, float *image, float *image_out)
{
    for (int h = 0; h < height; h++)
    {
        int offset_out = h * width;      // 1 color per pixel
        int offset     = offset_out * 3; // 3 colors per pixel
        
        for (int w = 0; w < width; w++)
        {
            float *pixel = &image[offset + w * 3];
            
            // Convert to grayscale following the "luminance" model
            image_out[offset_out + w] = pixel[0] * 0.0722f + // B
                                        pixel[1] * 0.7152f + // G
                                        pixel[2] * 0.2126f;  // R
        }
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (index_x < width && index_y < height)
    {
        int offset_out = index_y * width + index_x;
        int offset_in  = offset_out * 3;
        
        image_out[offset_out] = image[offset_in + 0] * 0.0722f + // B
                                image[offset_in + 1] * 0.7152f + // G
                                image[offset_in + 2] * 0.2126f;  // R
    }
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the CPU.
 */
float cpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    float pixel = 0.0f;
    
    for (int h = 0; h < filter_dim; h++)
    {
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    
    return pixel;
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the GPU.
 */
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    float pixel = 0.0f;                                                                                            
    for (int h = 0; h < filter_dim; h++) {                                                                         
        for (int w = 0; w < filter_dim; w++) {                                                                     
            pixel += image[h * stride + w] * matrix[h * filter_dim + w];                                           
        }
    }
    return pixel;
    ////////////////
    // TO-DO #5.2 ////////////////////////////////////////////////
    // Implement the GPU version of cpu_applyFilter()           //
    //                                                          //
    // Does it make sense to have a separate gpu_applyFilter()? //
    //////////////////////////////////////////////////////////////
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the CPU.
 */
void cpu_gaussian(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            image_out[offset + (w + 1)] = cpu_applyFilter(&image[offset_t + w],
                                                          width, gaussian, 3);
        }
    }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU.
 */
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (index_x < (width - 2) && index_y < (height - 2))
    {
        int offset_t = index_y * width + index_x;
        int offset   = (index_y + 1) * width + (index_x + 1);
        
        image_out[offset] = gpu_applyFilter(&image[offset_t],
                                            width, gaussian, 3);
    }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU (Shared Memory).
 */
__global__ void gpu_gaussian_shared(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    
    int gx = bx + tx;
    int gy = by + ty;
    
    tile[ty][tx] = (gx < width && gy < height) ? image[gy * width + gx] : 0.0f;
    
    if (tx < 2)
    {
        int gx2 = bx + tx + BLOCK_SIZE;
        tile[ty][tx + BLOCK_SIZE] = (gx2 < width && gy < height) ? image[gy * width + gx2] : 0.0f;
    }
    
    if (ty < 2)
    {
        int gy2 = by + ty + BLOCK_SIZE;
        tile[ty + BLOCK_SIZE][tx] = (gx < width && gy2 < height) ? image[gy2 * width + gx] : 0.0f;
    }
    
    if (tx < 2 && ty < 2)
    {
        int gx2 = bx + tx + BLOCK_SIZE;
        int gy2 = by + ty + BLOCK_SIZE;
        tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = (gx2 < width && gy2 < height) ? image[gy2 * width + gx2] : 0.0f;
    }
    
    __syncthreads();
    
    if (gx < (width - 2) && gy < (height - 2))
    {
        float pixel = tile[ty + 0][tx + 0] * gaussian[0] +
                      tile[ty + 0][tx + 1] * gaussian[1] +
                      tile[ty + 0][tx + 2] * gaussian[2] +
                      tile[ty + 1][tx + 0] * gaussian[3] +
                      tile[ty + 1][tx + 1] * gaussian[4] +
                      tile[ty + 1][tx + 2] * gaussian[5] +
                      tile[ty + 2][tx + 0] * gaussian[6] +
                      tile[ty + 2][tx + 1] * gaussian[7] +
                      tile[ty + 2][tx + 2] * gaussian[8];
        int offset = (gy + 1) * width + (gx + 1);
        image_out[offset] = pixel;
    }
}


/**
 * Calculates the gradient of an image using a Sobel filter on the CPU.
 */
void cpu_sobel(int width, int height, float *image, float *image_out)
{
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            float gx = cpu_applyFilter(&image[offset_t + w], width, sobel_x, 3);
            float gy = cpu_applyFilter(&image[offset_t + w], width, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU.
 */
__global__ void gpu_sobel(int width, int height, float *image, float *image_out)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (index_x < (width - 2) && index_y < (height - 2))
    {
        float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                             2.0f,  0.0f, -2.0f,
                             1.0f,  0.0f, -1.0f };
        float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                             0.0f,  0.0f,  0.0f,
                            -1.0f, -2.0f, -1.0f };
                            
        int offset_t = index_y * width + index_x;
        int offset   = (index_y + 1) * width + (index_x + 1);
        
        float gx = gpu_applyFilter(&image[offset_t], width, sobel_x, 3);
        float gy = gpu_applyFilter(&image[offset_t], width, sobel_y, 3);
        
        image_out[offset] = sqrtf(gx * gx + gy * gy);
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU (Shared Memory).
 */
__global__ void gpu_sobel_shared(int width, int height, float *image, float *image_out)
{
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    
    int gx = bx + tx;
    int gy = by + ty;
    
    tile[ty][tx] = (gx < width && gy < height) ? image[gy * width + gx] : 0.0f;
    
    if (tx < 2)
    {
        int gx2 = bx + tx + BLOCK_SIZE;
        tile[ty][tx + BLOCK_SIZE] = (gx2 < width && gy < height) ? image[gy * width + gx2] : 0.0f;
    }
    
    if (ty < 2)
    {
        int gy2 = by + ty + BLOCK_SIZE;
        tile[ty + BLOCK_SIZE][tx] = (gx < width && gy2 < height) ? image[gy2 * width + gx] : 0.0f;
    }
    
    if (tx < 2 && ty < 2)
    {
        int gx2 = bx + tx + BLOCK_SIZE;
        int gy2 = by + ty + BLOCK_SIZE;
        tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = (gx2 < width && gy2 < height) ? image[gy2 * width + gx2] : 0.0f;
    }
    
    __syncthreads();
    
    if (gx < (width - 2) && gy < (height - 2))
    {
        float gx_val = tile[ty + 0][tx + 0] - tile[ty + 0][tx + 2] +
                       2.0f * (tile[ty + 1][tx + 0] - tile[ty + 1][tx + 2]) +
                       tile[ty + 2][tx + 0] - tile[ty + 2][tx + 2];
                       
        float gy_val = tile[ty + 0][tx + 0] + 2.0f * tile[ty + 0][tx + 1] + tile[ty + 0][tx + 2] -
                       (tile[ty + 2][tx + 0] + 2.0f * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2]);
                       
        int offset = (gy + 1) * width + (gx + 1);
        image_out[offset] = sqrtf(gx_val * gx_val + gy_val * gy_val);
    }
}

int main(int argc, char **argv)
{
    BMPImage bitmap          = { 0 };
    float    *d_bitmap       = { 0 };
    float    *image_out_cpu[2] = { 0 };
    float    *image_out_gpu[2] = { 0 };
    float    *image_out_gpu_shared[2] = { 0 };
    float    *d_image_out[2]   = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    double   elapsed[2]      = { 0 };
    double   cpu_time_grayscale = 0.0;
    double   cpu_time_gaussian  = 0.0;
    double   cpu_time_sobel     = 0.0;
    double   elapsed_pipeline = 0.0;
    double   elapsed_cpu_pipeline = 0.0;
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.width * bitmap.height;
    grid       = dim3(((bitmap.width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));
    
    printf("Image opened (width=%d height=%d).\n", bitmap.width, bitmap.height);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++)
    {
        image_out_cpu[i] = (float *)calloc(image_size, sizeof(float));
        image_out_gpu[i] = (float *)calloc(image_size, sizeof(float));
        image_out_gpu_shared[i] = (float *)calloc(image_size, sizeof(float));
        
        gpuErrchk( cudaMalloc(&d_image_out[i], image_size * sizeof(float)) );
        gpuErrchk( cudaMemset(d_image_out[i], 0, image_size * sizeof(float)) );
    }

    gpuErrchk( cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3) );
    gpuErrchk( cudaMemcpy(d_bitmap, bitmap.data,
                          image_size * sizeof(float) * 3, cudaMemcpyHostToDevice) );
    
    // Step 1: Convert to grayscale
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_grayscale(bitmap.width, bitmap.height, bitmap.data, image_out_cpu[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        cpu_time_grayscale = elapsed[0];
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height,
                                       d_bitmap, d_image_out[0]);
        gpuErrchk( cudaGetLastError() );
        
        gpuErrchk( cudaMemcpy(image_out_gpu[0], d_image_out[0],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image in grayscale
        store_result(1, elapsed[0], elapsed[1], bitmap.width, bitmap.height,
                     image_out_cpu[0], image_out_gpu[0], 1, "");
    }
    
    // Step 2: Apply a 3x3 Gaussian filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_gaussian(bitmap.width, bitmap.height, image_out_cpu[0], image_out_cpu[1]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        cpu_time_gaussian = elapsed[0];
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_gaussian<<<grid, block>>>(bitmap.width, bitmap.height,
                                      d_image_out[0], d_image_out[1]);
        gpuErrchk( cudaGetLastError() );
        
        gpuErrchk( cudaMemcpy(image_out_gpu[1], d_image_out[1],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image with the Gaussian filter applied
        store_result(2, elapsed[0], elapsed[1], bitmap.width, bitmap.height,
                     image_out_cpu[1], image_out_gpu[1], 1, "");

        // Launch the GPU version (Shared Memory)
        gpuErrchk( cudaMemset(d_image_out[1], 0, image_size * sizeof(float)) );
        
        double elapsed_shared = 0.0;
        gettimeofday(&t[0], NULL);
        gpu_gaussian_shared<<<grid, block>>>(bitmap.width, bitmap.height,
                                             d_image_out[0], d_image_out[1]);
        gpuErrchk( cudaGetLastError() );
        
        gpuErrchk( cudaMemcpy(image_out_gpu_shared[1], d_image_out[1],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed_shared = get_elapsed(t[0], t[1]);
        
        // Store the result image for the shared memory version (omits CPU time logging)
        store_result(2, 0, elapsed_shared, bitmap.width, bitmap.height,
                     image_out_cpu[1], image_out_gpu_shared[1], 1, "_shared");
    }
    
    // Step 3: Apply a Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_sobel(bitmap.width, bitmap.height, image_out_cpu[1], image_out_cpu[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        cpu_time_sobel = elapsed[0];
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_sobel<<<grid, block>>>(bitmap.width, bitmap.height,
                                   d_image_out[1], d_image_out[0]);
        gpuErrchk( cudaGetLastError() );
        
        gpuErrchk( cudaMemcpy(image_out_gpu[0], d_image_out[0],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the final result image with the Sobel filter applied
        store_result(3, elapsed[0], elapsed[1], bitmap.width, bitmap.height,
                     image_out_cpu[0], image_out_gpu[0], 1, "");

        // Launch the GPU version (Shared Memory)
        gpuErrchk( cudaMemset(d_image_out[0], 0, image_size * sizeof(float)) );
        
        double elapsed_shared = 0.0;
        gettimeofday(&t[0], NULL);
        gpu_sobel_shared<<<grid, block>>>(bitmap.width, bitmap.height,
                                          d_image_out[1], d_image_out[0]);
        gpuErrchk( cudaGetLastError() );
        
        gpuErrchk( cudaMemcpy(image_out_gpu_shared[0], d_image_out[0],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed_shared = get_elapsed(t[0], t[1]);
        
        // Store the result image for the shared memory version (omits CPU time logging)
        store_result(3, 0, elapsed_shared, bitmap.width, bitmap.height,
                     image_out_cpu[0], image_out_gpu_shared[0], 1, "_shared");
    }

    // Step 4: Pipelined GPU execution (Grayscale -> Gaussian -> Sobel)
    {
        float *image_out_gpu_pipeline = (float *)calloc(image_size, sizeof(float));
        
        // Reset the device buffers to ensure clean measurement
        gpuErrchk( cudaMemset(d_image_out[0], 0, image_size * sizeof(float)) );
        gpuErrchk( cudaMemset(d_image_out[1], 0, image_size * sizeof(float)) );
        
        gettimeofday(&t[0], NULL);
        
        // 1. Convert to grayscale (reads d_bitmap, writes d_image_out[0])
        gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height,
                                       d_bitmap, d_image_out[0]);
        gpuErrchk( cudaGetLastError() );
        
        // 2. Apply Gaussian filter (reads d_image_out[0], writes d_image_out[1])
        gpu_gaussian<<<grid, block>>>(bitmap.width, bitmap.height,
                                      d_image_out[0], d_image_out[1]);
        gpuErrchk( cudaGetLastError() );
        
        // 3. Apply Sobel filter (reads d_image_out[1], writes d_image_out[0])
        gpu_sobel<<<grid, block>>>(bitmap.width, bitmap.height,
                                   d_image_out[1], d_image_out[0]);
        gpuErrchk( cudaGetLastError() );
        
        // 4. Copy ONLY the final Sobel result back to the host
        gpuErrchk( cudaMemcpy(image_out_gpu_pipeline, d_image_out[0],
                              image_size * sizeof(float), cudaMemcpyDeviceToHost) );
        gettimeofday(&t[1], NULL);
        
        elapsed_pipeline = get_elapsed(t[0], t[1]);
        
        printf("GPU Pipeline Completed:\n");
        printf("  Total GPU Pipeline Time: %fms (including final memcpy)\n", elapsed_pipeline);
        
        // Store the pipelined result image
        writeBMPGrayscale(bitmap.width, bitmap.height, image_out_gpu_pipeline, "images/hw3_result_gpu_pipeline.bmp");
        printf("  GPU Pipeline result stored in \"images/hw3_result_gpu_pipeline.bmp\"\n");
        
        free(image_out_gpu_pipeline);
    }

    // Step 5: Pipelined CPU execution (Grayscale -> Gaussian -> Sobel back-to-back)
    {
        float *cpu_pipeline_temp[2];
        cpu_pipeline_temp[0] = (float *)calloc(image_size, sizeof(float));
        cpu_pipeline_temp[1] = (float *)calloc(image_size, sizeof(float));
        
        gettimeofday(&t[0], NULL);
        
        // 1. Grayscale
        cpu_grayscale(bitmap.width, bitmap.height, bitmap.data, cpu_pipeline_temp[0]);
        
        // 2. Gaussian
        cpu_gaussian(bitmap.width, bitmap.height, cpu_pipeline_temp[0], cpu_pipeline_temp[1]);
        
        // 3. Sobel
        cpu_sobel(bitmap.width, bitmap.height, cpu_pipeline_temp[1], cpu_pipeline_temp[0]);
        
        gettimeofday(&t[1], NULL);
        
        elapsed_cpu_pipeline = get_elapsed(t[0], t[1]);
        
        printf("CPU Pipeline Completed:\n");
        printf("  Total CPU Pipeline Time: %fms\n", elapsed_cpu_pipeline);
        
        // Store the pipelined CPU result to verify correctness
        writeBMPGrayscale(bitmap.width, bitmap.height, cpu_pipeline_temp[0], "images/hw3_result_cpu_pipeline.bmp");
        printf("  CPU Pipeline result stored in \"images/hw3_result_cpu_pipeline.bmp\"\n");
        
        free(cpu_pipeline_temp[0]);
        free(cpu_pipeline_temp[1]);
    }

    printf("\n==================================================\n");
    printf("          Pipeline Performance Summary            \n");
    printf("==================================================\n");
    printf("CPU Pipeline (Warm Caches, 1 block):   %f ms\n", elapsed_cpu_pipeline);
    printf("GPU Pipeline (Back-to-back, 1 copy):   %f ms\n", elapsed_pipeline);
    printf("==================================================\n\n");
    
    // Release the allocated memory
    for (int i = 0; i < 2; i++)
    {
        free(image_out_cpu[i]);
        free(image_out_gpu[i]);
        free(image_out_gpu_shared[i]);
        gpuErrchk( cudaFree(d_image_out[i]) );
    }
    
    freeBMP(bitmap);
    gpuErrchk( cudaFree(d_bitmap) );
    
    return 0;
}

