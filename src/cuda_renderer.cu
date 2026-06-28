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

struct cuda_vec3 {
    double x;
    double y;
    double z;
};

struct cuda_ray {
    cuda_vec3 origin;
    cuda_vec3 direction;
};

struct cuda_sphere {
    cuda_vec3 center;
    double radius;
};

__host__ __device__ cuda_vec3 make_vec3(double x, double y, double z) {
    return cuda_vec3{x, y, z};
}

__host__ __device__ cuda_vec3 operator+(const cuda_vec3& a, const cuda_vec3& b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ cuda_vec3 operator-(const cuda_vec3& a, const cuda_vec3& b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ cuda_vec3 operator*(double t, const cuda_vec3& v) {
    return make_vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ cuda_vec3 operator/(const cuda_vec3& v, double t) {
    return (1.0 / t) * v;
}

__host__ __device__ double dot(const cuda_vec3& a, const cuda_vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ double length(const cuda_vec3& v) {
    return sqrt(dot(v, v));
}

__host__ __device__ cuda_vec3 unit_vector(const cuda_vec3& v) {
    return v / length(v);
}

__host__ __device__ cuda_vec3 ray_at(const cuda_ray& ray, double t) {
    return ray.origin + t * ray.direction;
}

__host__ __device__ rgb to_rgb(const cuda_vec3& color) {
    return rgb{
        static_cast<unsigned char>(255.999 * color.x),
        static_cast<unsigned char>(255.999 * color.y),
        static_cast<unsigned char>(255.999 * color.z)
    };
}

__device__ bool hit_sphere(const cuda_sphere& sphere, const cuda_ray& ray, double t_min, double t_max, double& root) {
    const cuda_vec3 oc = ray.origin - sphere.center;
    const double a = dot(ray.direction, ray.direction);
    const double half_b = dot(oc, ray.direction);
    const double c = dot(oc, oc) - sphere.radius * sphere.radius;
    const double discriminant = half_b * half_b - a * c;

    if (discriminant < 0)
        return false;

    const double sqrtd = sqrt(discriminant);
    root = (-half_b - sqrtd) / a;

    if (root <= t_min || root >= t_max) {
        root = (-half_b + sqrtd) / a;
        if (root <= t_min || root >= t_max)
            return false;
    }

    return true;
}

__device__ cuda_vec3 background_color(const cuda_ray& ray) {
    const cuda_vec3 unit_direction = unit_vector(ray.direction);
    const double a = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - a) * make_vec3(1.0, 1.0, 1.0) + a * make_vec3(0.5, 0.7, 1.0);
}

__global__ void background_gradient_kernel(rgb* pixels, int image_width, int image_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const int index = y * image_width + x;
    const double t = static_cast<double>(y) / (image_height - 1);
    const double r = (1.0 - t) * 0.5 + t * 1.0;
    const double g = (1.0 - t) * 0.7 + t * 1.0;
    const double b = 1.0;

    pixels[index] = rgb{
        static_cast<unsigned char>(255.999 * r),
        static_cast<unsigned char>(255.999 * g),
        static_cast<unsigned char>(255.999 * b)
    };
}

__global__ void single_sphere_kernel(rgb* pixels, int image_width, int image_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const double aspect_ratio = static_cast<double>(image_width) / image_height;
    const double viewport_height = 2.0;
    const double viewport_width = aspect_ratio * viewport_height;
    const double focal_length = 1.0;

    const cuda_vec3 origin = make_vec3(0, 0, 0);
    const cuda_vec3 horizontal = make_vec3(viewport_width, 0, 0);
    const cuda_vec3 vertical = make_vec3(0, viewport_height, 0);
    const cuda_vec3 lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - make_vec3(0, 0, focal_length);

    const double u = static_cast<double>(x) / (image_width - 1);
    const double v = static_cast<double>(image_height - 1 - y) / (image_height - 1);
    const cuda_ray ray{origin, lower_left_corner + u * horizontal + v * vertical - origin};
    const cuda_sphere sphere{make_vec3(0, 0, -1), 0.5};

    double t = 0.0;
    cuda_vec3 color;
    if (hit_sphere(sphere, ray, 0.001, 1.0e30, t)) {
        const cuda_vec3 normal = unit_vector(ray_at(ray, t) - sphere.center);
        color = 0.5 * make_vec3(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0);
    } else {
        color = background_color(ray);
    }

    pixels[y * image_width + x] = to_rgb(color);
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

    background_gradient_kernel<<<block_count, block_size>>>(device_pixels, image_width, image_height);

    bool ok = true;
    ok = ok && check_cuda(cudaGetLastError(), "background_gradient_kernel launch");
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

bool render_cuda_single_sphere(const char* output_path, int image_width, int image_height) {
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

    single_sphere_kernel<<<block_count, block_size>>>(device_pixels, image_width, image_height);

    bool ok = true;
    ok = ok && check_cuda(cudaGetLastError(), "single_sphere_kernel launch");
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
        << "CUDA single sphere render:\n"
        << "  output: " << output_path << '\n'
        << "  image: " << image_width << "x" << image_height << '\n'
        << "  time: " << elapsed.count() << " seconds\n";

    return true;
}
