#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring> // for memcpy

// ==========================================
// FORCE DISCRETE GPU (NVIDIA / AMD)
// ==========================================
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

const int WIDTH = 1024;
const int SIZE = WIDTH * WIDTH;
const size_t BYTES = SIZE * sizeof(float);

std::string loadFile(const char* filename) {
    std::ifstream file(filename);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// 1. OpenGL Implementation
double run_opengl(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    // Check which GPU is running
    const GLubyte* renderer = glGetString(GL_RENDERER);
    std::cout << "   OpenGL Device: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;

    std::string sourceStr = loadFile("compute.glsl");
    const char* src = sourceStr.c_str();
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    // Quick error check
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "   [GL Error] Shader Compilation:\n" << infoLog << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glUseProgram(program);

    GLuint ssboA, ssboB, ssboC;
    glGenBuffers(1, &ssboA); glGenBuffers(1, &ssboB); glGenBuffers(1, &ssboC);

    auto t1 = std::chrono::high_resolution_clock::now();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, A.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, B.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboB);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboC);
    glBufferData(GL_SHADER_STORAGE_BUFFER, BYTES, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboC);

    glUniform1i(glGetUniformLocation(program, "WIDTH"), WIDTH);
    glDispatchCompute(WIDTH / 32, WIDTH / 32, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if(ptr) memcpy(C.data(), ptr, BYTES);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    auto t2 = std::chrono::high_resolution_clock::now();
    
    glDeleteBuffers(1, &ssboA); glDeleteBuffers(1, &ssboB); glDeleteBuffers(1, &ssboC);
    glDeleteProgram(program);
    glDeleteShader(shader);

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

// 2. OpenCL Implementation
double run_opencl(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    std::cout << "   OpenCL Device: " << name << std::endl;

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    std::string sourceStr = loadFile("compute.cl");
    const char* src = sourceStr.c_str();
    size_t len = sourceStr.length();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, BYTES, C.data(), 0, NULL, NULL);

    auto t2 = std::chrono::high_resolution_clock::now();

    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

int main() {
    std::cout << "Matrix: " << WIDTH << "x" << WIDTH << std::endl;
    std::vector<float> A(SIZE, 1.0f);
    std::vector<float> B(SIZE, 1.0f);
    std::vector<float> C_GL(SIZE);
    std::vector<float> C_CL(SIZE);

    // Init OpenGL
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Hidden", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    std::cout << "Starting OpenGL..." << std::endl;
    double timeGL = run_opengl(A, B, C_GL);
    std::cout << "   Time: " << timeGL << " ms" << std::endl;

    glfwTerminate(); 

    // Init OpenCL
    std::cout << "Starting OpenCL..." << std::endl;
    double timeCL = run_opencl(A, B, C_CL);
    std::cout << "   Time: " << timeCL << " ms" << std::endl;

    std::cout << "---------------------------------" << std::endl;
    if (timeGL < timeCL) std::cout << "Winner: OpenGL (" << timeCL/timeGL << "x faster)" << std::endl;
    else                 std::cout << "Winner: OpenCL (" << timeGL/timeCL << "x faster)" << std::endl;

    return 0;
}