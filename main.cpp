#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>

const int WIDTH = 1024;
const int SIZE = WIDTH * WIDTH;
const size_t BYTES = SIZE * sizeof(float);

// --- Utils ---
std::string loadFile(const char* filename) {
    std::ifstream file(filename);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// 1. OpenGL Implementation (Context Dependent)
double run_opengl(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    const GLubyte* renderer = glGetString(GL_RENDERER);
    std::cout << ">> OpenGL Device: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;

    std::string sourceStr = loadFile("compute.glsl");
    const char* src = sourceStr.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Error: " << infoLog << std::endl;
        return -1.0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glUseProgram(program);

    GLuint ssbo[3];
    glGenBuffers(3, ssbo);

    auto t1 = std::chrono::high_resolution_clock::now();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, A.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, B.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo[2]);

    glUniform1i(glGetUniformLocation(program, "WIDTH"), WIDTH);
    glDispatchCompute(WIDTH / 32, WIDTH / 32, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if(ptr) memcpy(C.data(), ptr, BYTES);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    auto t2 = std::chrono::high_resolution_clock::now();
    
    glDeleteBuffers(3, ssbo);
    glDeleteProgram(program);
    glDeleteShader(shader);

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

// 2. OpenCL Implementation (Specific Device)
double run_opencl_device(cl_device_id device, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    cl_int err;
    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    std::cout << ">> OpenCL Device: " << name << std::endl;

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) { std::cerr << "Context Error" << std::endl; return -1.0; }
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    std::string sourceStr = loadFile("compute.cl");
    const char* src = sourceStr.c_str();
    size_t len = sourceStr.length();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &err);
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "CL Build Error: " << log.data() << std::endl;
        return -1.0;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);

    auto t1 = std::chrono::high_resolution_clock::now();

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BYTES, (void*)A.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BYTES, (void*)B.data(), &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BYTES, NULL, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &WIDTH);

    size_t globalSize[2] = { (size_t)WIDTH, (size_t)WIDTH };
    size_t localSize[2] = { 32, 32 };
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) { std::cerr << "Execute Error: " << err << std::endl; }
    
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, BYTES, C.data(), 0, NULL, NULL);

    auto t2 = std::chrono::high_resolution_clock::now();

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

int main() {
    std::cout << "Benchmarking " << WIDTH << "x" << WIDTH << " Matrix Mul" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<float> A(SIZE, 1.0f);
    std::vector<float> B(SIZE, 1.0f);
    std::vector<float> C(SIZE);

    // --- 1. RUN OPENGL (Current Context) ---
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Hidden", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    double glTime = run_opengl(A, B, C);
    std::cout << "   Time: " << glTime << " ms" << std::endl;
    glfwTerminate();

    std::cout << "----------------------------------------" << std::endl;

    // --- 2. RUN OPENCL (Iterate ALL Platforms & Devices) ---
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    for (const auto& platform : platforms) {
        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        for (const auto& device : devices) {
            double clTime = run_opencl_device(device, A, B, C);
            std::cout << "   Time: " << clTime << " ms" << std::endl;
            std::cout << "   Speedup vs GL (Current): " << glTime / clTime << "x" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }

    return 0;
}
