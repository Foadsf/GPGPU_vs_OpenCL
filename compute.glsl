#version 430 core
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform int WIDTH;

layout(std430, binding = 0) buffer BufferA { float A[]; };
layout(std430, binding = 1) buffer BufferB { float B[]; };
layout(std430, binding = 2) buffer BufferC { float C[]; };

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    if (col >= WIDTH || row >= WIDTH) return;

    float sum = 0.0;
    for (int k = 0; k < WIDTH; k++) {
        sum += A[row * WIDTH + k] * B[k * WIDTH + col];
    }
    C[row * WIDTH + col] = sum;
}