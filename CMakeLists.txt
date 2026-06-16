cmake_minimum_required(VERSION 3.18)
project(RayTracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA renderer support when a CUDA Toolkit is available" ON)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()

        add_compile_definitions(RTWEEKEND_CUDA_ENABLED=1)
        message(STATUS "CUDA support enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "USE_CUDA is ON, but no CUDA compiler was found. Building the CPU renderer only.")
    endif()
endif()

add_executable(inOneWeekend
    src/main.cpp
    src/vec3.h
    src/color.h
    src/ray.h
    src/hittable.h
    src/sphere.h
    src/hittable_list.h
    src/rtweekend.h
    src/interval.h
    src/camera.h
    src/material.h
)
