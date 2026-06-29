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
    cuda_vec3 albedo;
};

struct hit_record {
    double t;
    cuda_vec3 point;
    cuda_vec3 normal;
    cuda_vec3 albedo;
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

__host__ __device__ cuda_vec3 operator*(const cuda_vec3& a, const cuda_vec3& b) {
    return make_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ cuda_vec3& operator+=(cuda_vec3& a, const cuda_vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
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

__host__ __device__ bool near_zero(const cuda_vec3& v) {
    const double epsilon = 1.0e-8;
    return fabs(v.x) < epsilon && fabs(v.y) < epsilon && fabs(v.z) < epsilon;
}

__host__ __device__ double clamp(double x, double min_value, double max_value) {
    if (x < min_value)
        return min_value;
    if (x > max_value)
        return max_value;
    return x;
}

__host__ __device__ cuda_vec3 ray_at(const cuda_ray& ray, double t) {
    return ray.origin + t * ray.direction;
}

__host__ __device__ rgb to_rgb(const cuda_vec3& color, double scale = 1.0) {
    const double r = sqrt(scale * color.x);
    const double g = sqrt(scale * color.y);
    const double b = sqrt(scale * color.z);

    return rgb{
        static_cast<unsigned char>(256 * clamp(r, 0.0, 0.999)),
        static_cast<unsigned char>(256 * clamp(g, 0.0, 0.999)),
        static_cast<unsigned char>(256 * clamp(b, 0.0, 0.999))
    };
}

__device__ unsigned int xorshift32(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ double random_double(unsigned int& state) {
    return (xorshift32(state) + 0.5) / 4294967296.0;
}

__device__ cuda_vec3 random_in_unit_sphere(unsigned int& state) {
    while (true) {
        const cuda_vec3 p = make_vec3(
            2.0 * random_double(state) - 1.0,
            2.0 * random_double(state) - 1.0,
            2.0 * random_double(state) - 1.0
        );

        if (dot(p, p) < 1.0)
            return p;
    }
}

__device__ cuda_vec3 random_unit_vector(unsigned int& state) {
    return unit_vector(random_in_unit_sphere(state));
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

__device__ bool hit_sphere_list(
    const cuda_sphere* spheres,
    int sphere_count,
    const cuda_ray& ray,
    double t_min,
    double t_max,
    hit_record& closest_hit
) {
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (int i = 0; i < sphere_count; i++) {
        double root = 0.0;
        if (hit_sphere(spheres[i], ray, t_min, closest_so_far, root)) {
            hit_anything = true;
            closest_so_far = root;
            closest_hit.t = root;
            closest_hit.point = ray_at(ray, root);
            closest_hit.normal = unit_vector(closest_hit.point - spheres[i].center);
            closest_hit.albedo = spheres[i].albedo;
        }
    }

    return hit_anything;
}

__device__ cuda_vec3 background_color(const cuda_ray& ray) {
    const cuda_vec3 unit_direction = unit_vector(ray.direction);
    const double a = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - a) * make_vec3(1.0, 1.0, 1.0) + a * make_vec3(0.5, 0.7, 1.0);
}

__device__ cuda_vec3 shade_with_simple_light(const hit_record& rec) {
    const cuda_vec3 light_direction = unit_vector(make_vec3(-1.0, 1.0, 0.5));
    const double diffuse = fmax(0.0, dot(rec.normal, light_direction));
    const double ambient = 0.25;
    return (ambient + 0.75 * diffuse) * rec.albedo;
}

__device__ cuda_ray get_simple_camera_ray(double u, double v, int image_width, int image_height);

__device__ cuda_ray get_simple_camera_ray(int x, int y, int image_width, int image_height) {
    const double u = static_cast<double>(x) / (image_width - 1);
    const double v = static_cast<double>(image_height - 1 - y) / (image_height - 1);
    return get_simple_camera_ray(u, v, image_width, image_height);
}

__device__ cuda_ray get_simple_camera_ray(double u, double v, int image_width, int image_height) {
    const double aspect_ratio = static_cast<double>(image_width) / image_height;
    const double viewport_height = 2.0;
    const double viewport_width = aspect_ratio * viewport_height;
    const double focal_length = 1.0;

    const cuda_vec3 origin = make_vec3(0, 0, 0);
    const cuda_vec3 horizontal = make_vec3(viewport_width, 0, 0);
    const cuda_vec3 vertical = make_vec3(0, viewport_height, 0);
    const cuda_vec3 lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - make_vec3(0, 0, focal_length);

    return cuda_ray{origin, lower_left_corner + u * horizontal + v * vertical - origin};
}

__device__ cuda_vec3 path_traced_color(
    cuda_ray ray,
    const cuda_sphere* spheres,
    int sphere_count,
    int max_depth,
    unsigned int& rng_state
) {
    cuda_vec3 attenuation = make_vec3(1.0, 1.0, 1.0);

    for (int depth = 0; depth < max_depth; depth++) {
        hit_record rec{};

        if (!hit_sphere_list(spheres, sphere_count, ray, 0.001, 1.0e30, rec))
            return attenuation * background_color(ray);

        cuda_vec3 scatter_direction = rec.normal + random_unit_vector(rng_state);
        if (near_zero(scatter_direction))
            scatter_direction = rec.normal;

        attenuation = attenuation * rec.albedo;
        ray = cuda_ray{rec.point, scatter_direction};
    }

    return make_vec3(0.0, 0.0, 0.0);
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

    const cuda_ray ray = get_simple_camera_ray(x, y, image_width, image_height);
    const cuda_sphere sphere{make_vec3(0, 0, -1), 0.5, make_vec3(0.7, 0.3, 0.3)};

    double t = 0.0;
    cuda_vec3 color;
    if (hit_sphere(sphere, ray, 0.001, 1.0e30, t)) {
        const cuda_vec3 normal = unit_vector(ray_at(ray, t) - sphere.center);
        const hit_record rec{t, ray_at(ray, t), normal, sphere.albedo};
        color = shade_with_simple_light(rec);
    } else {
        color = background_color(ray);
    }

    pixels[y * image_width + x] = to_rgb(color);
}

__global__ void multiple_spheres_kernel(
    rgb* pixels,
    int image_width,
    int image_height,
    const cuda_sphere* spheres,
    int sphere_count
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    const cuda_ray ray = get_simple_camera_ray(x, y, image_width, image_height);
    hit_record rec{};
    cuda_vec3 color;

    if (hit_sphere_list(spheres, sphere_count, ray, 0.001, 1.0e30, rec)) {
        color = shade_with_simple_light(rec);
    } else {
        color = background_color(ray);
    }

    pixels[y * image_width + x] = to_rgb(color);
}

__global__ void path_traced_spheres_kernel(
    rgb* pixels,
    int image_width,
    int image_height,
    const cuda_sphere* spheres,
    int sphere_count,
    int samples_per_pixel,
    int max_depth
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    unsigned int rng_state =
        2166136261u ^
        static_cast<unsigned int>(x + 1) * 16777619u ^
        static_cast<unsigned int>(y + 1) * 374761393u ^
        static_cast<unsigned int>(samples_per_pixel) * 668265263u;

    cuda_vec3 color = make_vec3(0.0, 0.0, 0.0);

    for (int sample = 0; sample < samples_per_pixel; sample++) {
        const double u = (x + random_double(rng_state)) / (image_width - 1);
        const double v = (image_height - 1 - y + random_double(rng_state)) / (image_height - 1);
        const cuda_ray ray = get_simple_camera_ray(u, v, image_width, image_height);
        color += path_traced_color(ray, spheres, sphere_count, max_depth, rng_state);
    }

    pixels[y * image_width + x] = to_rgb(color, 1.0 / samples_per_pixel);
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

bool render_cuda_multiple_spheres(const char* output_path, int image_width, int image_height) {
    const auto start = std::chrono::high_resolution_clock::now();
    const int pixel_count = image_width * image_height;
    const size_t pixel_bytes = pixel_count * sizeof(rgb);

    const std::vector<cuda_sphere> host_spheres{
        cuda_sphere{make_vec3(0, -100.5, -1), 100.0, make_vec3(0.55, 0.85, 0.35)},
        cuda_sphere{make_vec3(0, 0, -1), 0.5, make_vec3(0.7, 0.3, 0.3)},
        cuda_sphere{make_vec3(-1.0, 0, -1.2), 0.45, make_vec3(0.2, 0.35, 0.9)},
        cuda_sphere{make_vec3(1.0, 0, -1.2), 0.45, make_vec3(0.9, 0.75, 0.25)}
    };

    rgb* device_pixels = nullptr;
    cuda_sphere* device_spheres = nullptr;

    if (!check_cuda(cudaMalloc(&device_pixels, pixel_bytes), "cudaMalloc pixels"))
        return false;

    bool ok = true;
    ok = ok && check_cuda(
        cudaMalloc(&device_spheres, host_spheres.size() * sizeof(cuda_sphere)),
        "cudaMalloc spheres"
    );

    if (ok) {
        ok = check_cuda(
            cudaMemcpy(
                device_spheres,
                host_spheres.data(),
                host_spheres.size() * sizeof(cuda_sphere),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy spheres host to device"
        );
    }

    const dim3 block_size(16, 16);
    const dim3 block_count(
        (image_width + block_size.x - 1) / block_size.x,
        (image_height + block_size.y - 1) / block_size.y
    );

    if (ok) {
        multiple_spheres_kernel<<<block_count, block_size>>>(
            device_pixels,
            image_width,
            image_height,
            device_spheres,
            static_cast<int>(host_spheres.size())
        );

        ok = ok && check_cuda(cudaGetLastError(), "multiple_spheres_kernel launch");
        ok = ok && check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::vector<rgb> host_pixels(pixel_count);
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(host_pixels.data(), device_pixels, pixel_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy pixels device to host"
        );
    }

    check_cuda(cudaFree(device_spheres), "cudaFree spheres");
    check_cuda(cudaFree(device_pixels), "cudaFree pixels");

    if (!ok)
        return false;

    if (!write_ppm(output_path, host_pixels, image_width, image_height))
        return false;

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    std::clog
        << "CUDA multiple spheres render:\n"
        << "  output: " << output_path << '\n'
        << "  image: " << image_width << "x" << image_height << '\n'
        << "  spheres: " << host_spheres.size() << '\n'
        << "  time: " << elapsed.count() << " seconds\n";

    return true;
}

bool render_cuda_path_traced_spheres(
    const char* output_path,
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth
) {
    const auto start = std::chrono::high_resolution_clock::now();
    const int pixel_count = image_width * image_height;
    const size_t pixel_bytes = pixel_count * sizeof(rgb);

    const std::vector<cuda_sphere> host_spheres{
        cuda_sphere{make_vec3(0, -100.5, -1), 100.0, make_vec3(0.55, 0.85, 0.35)},
        cuda_sphere{make_vec3(0, 0, -1), 0.5, make_vec3(0.7, 0.3, 0.3)},
        cuda_sphere{make_vec3(-1.0, 0, -1.2), 0.45, make_vec3(0.2, 0.35, 0.9)},
        cuda_sphere{make_vec3(1.0, 0, -1.2), 0.45, make_vec3(0.9, 0.75, 0.25)}
    };

    rgb* device_pixels = nullptr;
    cuda_sphere* device_spheres = nullptr;

    if (!check_cuda(cudaMalloc(&device_pixels, pixel_bytes), "cudaMalloc pixels"))
        return false;

    bool ok = true;
    ok = ok && check_cuda(
        cudaMalloc(&device_spheres, host_spheres.size() * sizeof(cuda_sphere)),
        "cudaMalloc spheres"
    );

    if (ok) {
        ok = check_cuda(
            cudaMemcpy(
                device_spheres,
                host_spheres.data(),
                host_spheres.size() * sizeof(cuda_sphere),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy spheres host to device"
        );
    }

    const dim3 block_size(16, 16);
    const dim3 block_count(
        (image_width + block_size.x - 1) / block_size.x,
        (image_height + block_size.y - 1) / block_size.y
    );

    if (ok) {
        path_traced_spheres_kernel<<<block_count, block_size>>>(
            device_pixels,
            image_width,
            image_height,
            device_spheres,
            static_cast<int>(host_spheres.size()),
            samples_per_pixel,
            max_depth
        );

        ok = ok && check_cuda(cudaGetLastError(), "path_traced_spheres_kernel launch");
        ok = ok && check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::vector<rgb> host_pixels(pixel_count);
    if (ok) {
        ok = check_cuda(
            cudaMemcpy(host_pixels.data(), device_pixels, pixel_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy pixels device to host"
        );
    }

    check_cuda(cudaFree(device_spheres), "cudaFree spheres");
    check_cuda(cudaFree(device_pixels), "cudaFree pixels");

    if (!ok)
        return false;

    if (!write_ppm(output_path, host_pixels, image_width, image_height))
        return false;

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const auto total_samples =
        static_cast<long long>(image_width) * image_height * samples_per_pixel;

    std::clog
        << "CUDA path traced spheres render:\n"
        << "  output: " << output_path << '\n'
        << "  image: " << image_width << "x" << image_height << '\n'
        << "  spheres: " << host_spheres.size() << '\n'
        << "  samples_per_pixel: " << samples_per_pixel << '\n'
        << "  max_depth: " << max_depth << '\n'
        << "  total primary samples: " << total_samples << '\n'
        << "  time: " << elapsed.count() << " seconds\n"
        << "  primary samples/sec: " << (total_samples / elapsed.count()) << '\n';

    return true;
}
