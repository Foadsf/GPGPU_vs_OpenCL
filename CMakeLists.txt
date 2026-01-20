cmake_minimum_required(VERSION 3.15)
project(GPGPU_vs_OpenCL)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenCL REQUIRED) 

add_executable(gpgpu_vs_cl main.cpp)

# Link both GL and CL
target_link_libraries(gpgpu_vs_cl PRIVATE glfw glad::glad OpenCL::OpenCL)

set_property(TARGET gpgpu_vs_cl PROPERTY CXX_STANDARD 17)

# Copy both kernel files
add_custom_command(TARGET gpgpu_vs_cl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/compute.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/compute.cl"
        $<TARGET_FILE_DIR:gpgpu_vs_cl>
)