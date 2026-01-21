__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int width) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col < width && row < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}
