#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

bool render_cuda_gradient(const char* output_path, int image_width, int image_height);
bool render_cuda_single_sphere(const char* output_path, int image_width, int image_height);
bool render_cuda_multiple_spheres(const char* output_path, int image_width, int image_height);
bool render_cuda_path_traced_spheres(
    const char* output_path,
    int image_width,
    int image_height,
    int samples_per_pixel,
    int max_depth
);

#endif
