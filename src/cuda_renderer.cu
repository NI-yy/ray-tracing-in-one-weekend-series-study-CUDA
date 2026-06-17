#include "cuda_renderer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

struct rgb {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

__global__ void gradient_kernel(rgb* pixels, int image_width, int image_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const int index = y * image_width + x;
    const double r = static_cast<double>(x) / (image_width - 1);
    const double g = static_cast<double>(y) / (image_height - 1);

    pixels[index] = rgb{
        static_cast<unsigned char>(255.999 * r),
        static_cast<unsigned char>(255.999 * g),
        0
    };
}

bool check_cuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess)
        return true;

    std::cerr << "CUDA error during " << operation << ": "
              << cudaGetErrorString(result) << '\n';
    return false;
}

bool write_ppm(const char* output_path, const std::vector<rgb>& pixels, int image_width, int image_height) {
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Failed to open output file: " << output_path << '\n';
        return false;
    }

    output << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            const auto& pixel = pixels[y * image_width + x];
            output << static_cast<int>(pixel.r) << ' '
                   << static_cast<int>(pixel.g) << ' '
                   << static_cast<int>(pixel.b) << '\n';
        }
    }

    return true;
}

} // namespace

bool render_cuda_gradient(const char* output_path, int image_width, int image_height) {
    const auto start = std::chrono::high_resolution_clock::now();
    const int pixel_count = image_width * image_height;
    const size_t bytes = pixel_count * sizeof(rgb);

    rgb* device_pixels = nullptr;
    if (!check_cuda(cudaMalloc(&device_pixels, bytes), "cudaMalloc"))
        return false;

    const dim3 block_size(16, 16);
    const dim3 block_count(
        (image_width + block_size.x - 1) / block_size.x,
        (image_height + block_size.y - 1) / block_size.y
    );

    gradient_kernel<<<block_count, block_size>>>(device_pixels, image_width, image_height);

    bool ok = true;
    ok = ok && check_cuda(cudaGetLastError(), "gradient_kernel launch");
    ok = ok && check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<rgb> host_pixels(pixel_count);
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(host_pixels.data(), device_pixels, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host"
        );
    }

    check_cuda(cudaFree(device_pixels), "cudaFree");

    if (!ok)
        return false;

    if (!write_ppm(output_path, host_pixels, image_width, image_height))
        return false;

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::clog
        << "CUDA gradient render:\n"
        << "  output: " << output_path << '\n'
        << "  image: " << image_width << "x" << image_height << '\n'
        << "  time: " << elapsed.count() << " seconds\n";

    return true;
}
