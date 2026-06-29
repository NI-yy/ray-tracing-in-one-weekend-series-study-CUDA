#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#ifdef RTWEEKEND_CUDA_ENABLED
#include "cuda_renderer.h"
#endif

int main() {
#ifdef RTWEEKEND_CUDA_ENABLED
    render_cuda_path_traced_spheres("image_cuda_path_traced.ppm", 200, 112, 20, 10);
#endif

    hittable_list world;

    world.add(make_shared<sphere>(
        point3(0, -100.5, -1),
        100.0,
        make_shared<lambertian>(color(0.55, 0.85, 0.35))
    ));
    world.add(make_shared<sphere>(
        point3(0, 0, -1),
        0.5,
        make_shared<lambertian>(color(0.7, 0.3, 0.3))
    ));
    world.add(make_shared<sphere>(
        point3(-1.0, 0, -1.2),
        0.45,
        make_shared<lambertian>(color(0.2, 0.35, 0.9))
    ));
    world.add(make_shared<sphere>(
        point3(1.0, 0, -1.2),
        0.45,
        make_shared<lambertian>(color(0.9, 0.75, 0.25))
    ));

    camera cam;

    cam.aspect_ratio = 200.0 / 112.0;
    cam.image_width = 200;
    cam.samples_per_pixel = 20;
    cam.max_depth = 10;
    cam.vfov = 90;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0.0;
    cam.focus_dist = 1.0;

    cam.render(world, "image_cpu_comparison.ppm");
}
