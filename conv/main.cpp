

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

// Length of our convolution mask
#define MASK_LENGTH 7


// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }

    // std::cout<<"CPU: "<<temp<<" GPU: "<<result[i]<<" i:  "<<i<<"\n";
    // std::cout<<result[i+1]<<"\n";
    assert(temp == result[i]);
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
std::string kernelFile = "./include/1d_tiled.cl";

// unsigned short data[NUM_SHORTS];


int main() {

    // Load radix_kernel solver
    std::string kernel_code = loadKernelSource(kernelFile);

    //If there are no opencl platforms -  all_platforms == 0 and the program exits. 

    //One of the key features of OpenCL is its portability. So, for instance, there might be situations
    // in which both the CPU and the GPU can run OpenCL code. Thus, 
    // a good practice is to verify the OpenCL platforms to choose on which the compiled code run.

    // Number of elements in result array
    int n = 1 << 20;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Size of the mask in bytes
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    // Radius for padding the array
    int r = MASK_LENGTH / 2;
    int n_p = n + r * 2;

    // Size of the padded array in bytes
    size_t bytes_p = n_p * sizeof(int);

    // Allocate the array (include edge elements)...
    int *h_array = new int[n_p];

    // ... and initialize it
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
        h_array[i] = 0;
        } else {
        h_array[i] = rand() % 100;
        }
    }

    // Allocate the mask and initialize it
    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate space for the result
    int *h_result = new int[n];

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


    // create buffers on the device
    cl::Buffer array_d(context, CL_MEM_READ_ONLY, bytes_p);
    cl::Buffer result_d(context, CL_MEM_WRITE_ONLY, bytes_n);
    cl::Buffer mask_d(context, CL_MEM_READ_ONLY, bytes_m);
    

    // Create and write data to buffer
    queue.enqueueWriteBuffer(array_d, CL_TRUE, 0, bytes_p, (void *)h_array);
    queue.enqueueWriteBuffer(mask_d, CL_TRUE, 0, bytes_m, (void *)h_mask);



    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    //If runtime compilation are found they are presented in this point of the program.

    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = n;

    // Amount of space per-block for shared memory
    // This is padded by the overhanging radius on either side
    size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    // For Profiling
    cl::Event event;

    cl::Kernel kernel(program, "convolution_1d");
    // cl::Kernel kernel(program, "tiledMultiply");
    kernel.setArg(0, array_d);
    kernel.setArg(1, result_d);
    kernel.setArg(2, cl::Local(SHMEM));
    kernel.setArg(3, mask_d);
    
    //Details to enqueue the kernel for execution.

    cl::NDRange globalSize(GRID);
    cl::NDRange localSize(THREADS);

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);

    queue.finish();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    //read result from the device to array h_result
    queue.enqueueReadBuffer(result_d, CL_TRUE, 0, bytes_n, h_result);

    // Verify the result
    verify_result(h_array, h_mask, h_result, n);

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

    // Free allocated memory on the device and host
    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    return 0;
}
