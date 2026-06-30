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
        point3(0, -1000, 0),
        1000.0,
        make_shared<lambertian>(color(0.5, 0.5, 0.5))
    ));

    unsigned int rng_state = 0x12345678u;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            const point3 center(
                a + 0.9 * demo_random_double(rng_state),
                0.2,
                b + 0.9 * demo_random_double(rng_state)
            );

            if ((center - point3(4, 0.2, 0)).length_squared() <= 0.81)
                continue;

            const double choose_material = demo_random_double(rng_state);
            if (choose_material < 0.8) {
                const color albedo(
                    demo_random_double(rng_state) * demo_random_double(rng_state),
                    demo_random_double(rng_state) * demo_random_double(rng_state),
                    demo_random_double(rng_state) * demo_random_double(rng_state)
                );
                world.add(make_shared<sphere>(center, 0.2, make_shared<lambertian>(albedo)));
            } else if (choose_material < 0.95) {
                const color albedo(
                    demo_random_double(rng_state, 0.5, 1.0),
                    demo_random_double(rng_state, 0.5, 1.0),
                    demo_random_double(rng_state, 0.5, 1.0)
                );
                const double fuzz = demo_random_double(rng_state, 0.0, 0.5);
                world.add(make_shared<sphere>(center, 0.2, make_shared<metal>(albedo, fuzz)));
            } else {
                world.add(make_shared<sphere>(center, 0.2, make_shared<dielectric>(1.5)));
            }
        }
    }

    world.add(make_shared<sphere>(
        point3(0, 1, 0),
        1.0,
        make_shared<dielectric>(1.5)
    ));
    world.add(make_shared<sphere>(
        point3(-4, 1, 0),
        1.0,
        make_shared<lambertian>(color(0.4, 0.2, 0.1))
    ));
    world.add(make_shared<sphere>(
        point3(4, 1, 0),
        1.0,
        make_shared<metal>(color(0.7, 0.6, 0.5), 0.0)
    ));
}

} // namespace

int main() {
#ifdef RTWEEKEND_CUDA_ENABLED
    render_cuda_path_traced_spheres("image_cuda_path_traced_full.ppm", 1200, 675, 500, 50);
    return 0;
#endif

    hittable_list world;
    add_demo_scene(world);

    camera cam;

    cam.aspect_ratio = 200.0 / 112.0;
    cam.image_width = 200;
    cam.samples_per_pixel = 5;
    cam.max_depth = 10;
    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    cam.render(world, "image_cpu_comparison.ppm");
}
