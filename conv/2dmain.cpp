

//Original tutorial: "Tutorial: Simple start with OpenCL and C++", 
//https://programmerclick.com/article/47811146604/


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <random>
#include <cassert>
#include <cstdlib>

// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)


// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void verify_result(std::vector<int>&m, std::vector<int>&mask, std::vector<int>&result, int N) {
  // Temp value for accumulating results
  int temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int i = 0; i < N; i++) {
    // Go over each column
    for (int j = 0; j < N; j++) {
      // Reset the temp variable
      temp = 0;

      // Go over each mask row
      for (int k = 0; k < MASK_DIM; k++) {
        // Update offset value for row
        offset_r = i - MASK_OFFSET + k;

        // Go over each mask column
        for (int l = 0; l < MASK_DIM; l++) {
          // Update offset value for column
          offset_c = j - MASK_OFFSET + l;

          // Range checks if we are hanging off the matrix
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              // Accumulate partial results
              temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // Fail if the results don't match
      assert(result[i * N + j] == temp);
    }
  }
}

// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  n : Dimension of the matrix (square)
void init_matrix(std::vector<int>&m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}


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
std::string kernelFile = "./include/2d_const.cl";

// unsigned short data[NUM_SHORTS];


int main() {

    // Load radix_kernel solver
    std::string kernel_code = loadKernelSource(kernelFile);

    // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
    int N = 1 << 10;

    // Size of the array in bytes
    size_t bytes_n = N * N * sizeof(int);

    // Initialize matrix and result matrix
    std::vector<int> matrix_h(N*N);
    std::vector<int> result_h(N*N);
    init_matrix(matrix_h, N);
    
    // std::cout<<matrix_h.size()<<"\n";

    // Size of the mask in bytes
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

    // Allocate the mask and initialize it
    std::vector<int> mask_h(MASK_DIM*MASK_DIM);
    init_matrix(mask_h, MASK_DIM);

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    //We are going to use the platform of id == 0
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

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

    // Define properties for the command queue
    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };

    //create context, kernel source and queue to push commands to the device.
    cl::Context context({ default_device });
    // cl::CommandQueue queue(context, default_device);
    cl::CommandQueue queue(context, default_device, properties); // For profiling
    cl::Program::Sources sources;

    //Appending the kernel, which is presented here as a string. 
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    //OpenCL compiles the kernel in runtime, that's the reason it is expressed as a string. 
    //There are also ways to compile the device-side code offline. 
    cl::Program program(context, sources);


    // create buffers and write data on the device
    cl::Buffer matrix_d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_n, matrix_h.data());
    cl::Buffer result_d(context, CL_MEM_WRITE_ONLY, bytes_n);
    cl::Buffer mask_d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_m, mask_h.data());
    

    // Create and write data to buffer (ALTERNATE WAY)
    // queue.enqueueWriteBuffer(matrix_d, CL_TRUE, 0, bytes_n, (void *)matrix_h.data());
    // queue.enqueueWriteBuffer(mask_d, CL_TRUE, 0, bytes_m, (void *)mask_h.data());



    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }


    // Amount of space per-block for shared memory
    // This is padded by the overhanging radius on either side
    // size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    // For Profiling
    cl::Event event;

    cl::Kernel kernel(program, "convolution_2d");
    // cl::Kernel kernel(program, "tiledMultiply");
    kernel.setArg(0, matrix_d);
    kernel.setArg(1, result_d);
    kernel.setArg(2, mask_d);
    kernel.setArg(3, N);
    kernel.setArg(4, MASK_DIM);
    kernel.setArg(5, MASK_OFFSET);
    

    // Threads per TB
    int LOCAL = 16;

    // Dimension of Matrix
    int GLOBAL = ((N + LOCAL - 1) / LOCAL) * LOCAL;

    cl::NDRange globalSize(GLOBAL, GLOBAL);
    cl::NDRange localSize(LOCAL, LOCAL);

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);

    queue.finish();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //read result from the device to array h_result
    queue.enqueueReadBuffer(result_d, CL_TRUE, 0, bytes_n, result_h.data());

    // Verify the result
    verify_result(matrix_h, mask_h, result_h, N);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << elapsed_time << " nanoseconds" << std::endl;

    return 0;
}
