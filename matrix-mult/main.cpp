

//Original tutorial: "Tutorial: Simple start with OpenCL and C++", 
//https://programmerclick.com/article/47811146604/


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define NUM_SHORTS 8
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <random>


// static float A[8] = {
//   1.0f,  1.0f,  1.0f,  1.0f,
//   1.0f,  1.0f,  1.0f,  1.0f};


// static float A[8] = {
//   5.3f, 7.1f, 6.2f, 9.4f,
//   3.2f, 4.8f, 1.9f, 7.6f};

// // static float B[24] = {
// //   2.0f,  2.0f,  2.0f,  3.0f, 2.0f, 2.0f,
// //   2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f,
// //   2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f,
// //   2.0f,  2.0f,  2.0f,  2.0f, 2.0f, 2.0f};

// static float B[24] = {
//   2.5f, 1.3f, 4.8f, 3.9f, 2.7f, 1.8f,
//   7.4f, 5.2f, 8.1f, 9.2f, 6.1f, 5.6f,
//   1.9f, 3.5f, 7.3f, 6.8f, 2.2f, 4.7f,
//   8.4f, 7.6f, 2.3f, 4.1f, 5.5f, 3.3f};

std::vector<float> generateRandomMatrix(int rows, int cols) {
    std::vector<float> matrix(rows * cols);
    std::default_random_engine generator(static_cast<unsigned>(time(0)));
    std::uniform_real_distribution<float> distribution(1.0f, 10.0f);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = distribution(generator);
    }
    return matrix;
}

void printMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

using namespace std; 

std::string loadKernelSource(const std::string& fileName) {
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to load kernel." << std::endl;
        exit(1);
    }

    std::string sourceStr((std::istreambuf_iterator<char>(kernelFile)),
                           std::istreambuf_iterator<char>());
    kernelFile.close();

    return sourceStr;
}

// load kernel for radix_sort
std::string kernelFile = "./include/mm_kernel.cl";

// unsigned short data[NUM_SHORTS];


int main() {

    // Load radix_kernel solver
    std::string kernel_code = loadKernelSource(kernelFile);

    //If there are no opencl platforms -  all_platforms == 0 and the program exits. 

    //One of the key features of OpenCL is its portability. So, for instance, there might be situations
    // in which both the CPU and the GPU can run OpenCL code. Thus, 
    // a good practice is to verify the OpenCL platforms to choose on which the compiled code run.

    // Matrix dimensions
    // int wA=4;
    // int hA=2;
    // int wB=6;
    // int hB=4;
    int wA=1000;
    int hA=1000;
    int wB=1000;
    int hB=1000;
    int wC = wB;
    int hC = hA;
    std::vector<float> A = generateRandomMatrix(hA, wA);
    std::vector<float> B = generateRandomMatrix(hB, wB);
    std::vector<float> C(hC * wC, 0.0f);

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    //We are going to use the platform of id == 0
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";


    //An OpenCL platform might have several devices. 
    //The next step is to ensure that the code will run on the first device of the platform, 
    //if found. 

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    size_t max_work_group_size = default_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "Maximum work-group size: " << max_work_group_size << "\n";

    // std::cout << "Double FP = " << default_device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";

    //create context, kernel source and queue to push commands to the device.
    cl::Context context({ default_device });
    cl::CommandQueue queue(context, default_device);
    cl::Program::Sources sources;

    //Appending the kernel, which is presented here as a string. 
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    //OpenCL compiles the kernel in runtime, that's the reason it is expressed as a string. 
    //There are also ways to compile the device-side code offline. 
    cl::Program program(context, sources);


    // create buffers on the device
    // cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    // cl::Buffer C_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    // cl::Buffer D_d(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);

    // int A_h[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    // int C_h[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    

    //write arrays A and B to the device
    // queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    // queue.enqueueWriteBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

    // Create and write data to buffer
    cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(float)*wA*hA);
    queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(float)*wA*hA, (void *)A.data());

    cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(float)*wB*hB);
    queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(float)*wB*hB, (void *)B.data());

    cl::Buffer C_d(context, CL_MEM_WRITE_ONLY, sizeof(float)*wC*hC);



    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    //If runtime compilation are found they are presented in this point of the program.


    //From the program, which contains the "simple_add" kernel, create a kernel for execution
    //with three cl:buffers as parameters.
    //The types must match the arguments of the kernel function. 
    // cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
    cl::Kernel kernel(program, "simpleMultiply");
    // cl::Kernel kernel(program, "tiledMultiply");
    kernel.setArg(0, C_d);
    kernel.setArg(1, wA);
    kernel.setArg(2, hA);
    kernel.setArg(3, wB);
    kernel.setArg(4, hB);
    kernel.setArg(5, A_d);
    kernel.setArg(6, B_d);
    
    //Details to enqueue the kernel for execution.

    cl::NDRange globalSize(wC, hC);
    cl::NDRange localSize(20, 20);
    // cl::NDRange localSize(2, 2);
    // cl::NDRange globalSize((wC + 15) / 16 * 16, (hC + 15) / 16 * 16);
    // cl::NDRange localSize(16, 16);

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);

    queue.finish();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // unsigned short D_h[NUM_SHORTS];
    //read result C_d from the device to array C_h
    queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(float)*wC*hC, C.data());

    // std::cout << "Result: \n";
    // for (int i = 0; i<wC*hC; i++) {
    //     std::cout << C[i] << " ";
    // }

    std::cout << "\n";

    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    return 0;
}
