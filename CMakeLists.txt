cmake_minimum_required(VERSION 3.10)
project(RayTracing)

# C++11標準を使用するように設定
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 実行ファイルの設定（src/main.cpp をコンパイルする）
add_executable(inOneWeekend src/main.cpp "src/vec3.h" "src/color.h" "src/ray.h" "src/hittable.h" "src/sphere.h" "src/hittable_list.h" "src/rtweekend.h" "src/interval.h" "src/camera.h" "src/material.h" "src/aabb.h" "src/bvh.h" "src/texture.h" "src/rtw_stb_image.h" "src/perlin.h" "src/quad.h" "src/constant_medium.h" "src/onb.h" "src/pdf.h")