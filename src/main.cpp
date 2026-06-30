#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#ifdef RTWEEKEND_CUDA_ENABLED
#include "cuda_renderer.h"
#endif

namespace {

double demo_random_double(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state + 0.5) / 4294967296.0;
}

double demo_random_double(unsigned int& state, double min_value, double max_value) {
    return min_value + (max_value - min_value) * demo_random_double(state);
}

void add_demo_scene(hittable_list& world) {
    world.add(make_shared<sphere>(
        point3(0, -100.5, -1),
        100.0,
        make_shared<lambertian>(color(0.55, 0.85, 0.35))
    ));
    world.add(make_shared<sphere>(
        point3(0, 0, -1),
        0.5,
        make_shared<dielectric>(1.5)
    ));
    world.add(make_shared<sphere>(
        point3(-1.0, 0, -1.2),
        0.45,
        make_shared<metal>(color(0.2, 0.35, 0.9), 0.15)
    ));
    world.add(make_shared<sphere>(
        point3(1.0, 0, -1.2),
        0.45,
        make_shared<metal>(color(0.9, 0.75, 0.25), 0.05)
    ));

    unsigned int rng_state = 0x12345678u;
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 10; column++) {
            const double radius = demo_random_double(rng_state, 0.07, 0.13);
            const point3 center(
                -1.8 + column * 0.4 + demo_random_double(rng_state, -0.08, 0.08),
                -0.5 + radius,
                -0.65 - row * 0.32 + demo_random_double(rng_state, -0.08, 0.08)
            );

            if ((center - point3(0, 0, -1)).length_squared() < 0.55 ||
                (center - point3(-1.0, 0, -1.2)).length_squared() < 0.42 ||
                (center - point3(1.0, 0, -1.2)).length_squared() < 0.42)
                continue;

            const double choose_material = demo_random_double(rng_state);
            if (choose_material < 0.65) {
                const color albedo(
                    demo_random_double(rng_state, 0.15, 0.95) * demo_random_double(rng_state, 0.15, 0.95),
                    demo_random_double(rng_state, 0.15, 0.95) * demo_random_double(rng_state, 0.15, 0.95),
                    demo_random_double(rng_state, 0.15, 0.95) * demo_random_double(rng_state, 0.15, 0.95)
                );
                world.add(make_shared<sphere>(center, radius, make_shared<lambertian>(albedo)));
            } else if (choose_material < 0.9) {
                const color albedo(
                    demo_random_double(rng_state, 0.5, 1.0),
                    demo_random_double(rng_state, 0.5, 1.0),
                    demo_random_double(rng_state, 0.5, 1.0)
                );
                const double fuzz = demo_random_double(rng_state, 0.0, 0.35);
                world.add(make_shared<sphere>(center, radius, make_shared<metal>(albedo, fuzz)));
            } else {
                world.add(make_shared<sphere>(center, radius, make_shared<dielectric>(1.5)));
            }
        }
    }
}

} // namespace

int main() {
#ifdef RTWEEKEND_CUDA_ENABLED
    render_cuda_path_traced_spheres("image_cuda_path_traced.ppm", 200, 112, 20, 10);
#endif

    hittable_list world;
    add_demo_scene(world);

    camera cam;

    cam.aspect_ratio = 200.0 / 112.0;
    cam.image_width = 200;
    cam.samples_per_pixel = 20;
    cam.max_depth = 10;
    cam.vfov = 90;
    cam.lookfrom = point3(0, 0, 0);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 2.0;
    cam.focus_dist = 1.0;

    cam.render(world, "image_cpu_comparison.ppm");
}
