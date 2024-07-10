

//Original tutorial: "Tutorial: Simple start with OpenCL and C++", 
//https://programmerclick.com/article/47811146604/


#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define NUM_SHORTS 8
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

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
std::string kernelFile = "./include/radix_kernel.cl";

unsigned short data[NUM_SHORTS];


int main() {

    // Load radix_kernel solver
    std::string kernel_code = loadKernelSource(kernelFile);

    // Generate random input for radix sort
    // Initialize random seed
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    unsigned seed = static_cast<unsigned>(duration.count());
    std::srand(seed);

    // Initialize data
    for (size_t i = 0; i < NUM_SHORTS; ++i) {
        data[i] = static_cast<unsigned short>(i);
    }

    // Shuffle data
    for (size_t i = 0; i < NUM_SHORTS - 1; ++i) {
        size_t j = i + (std::rand() % (NUM_SHORTS - i));
        std::swap(data[i], data[j]);
    }

    // Print input
    std::cout << "Input: " << std::endl;
    for (size_t i = 0; i < NUM_SHORTS; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "" << "\n";

    //If there are no opencl platforms -  all_platforms == 0 and the program exits. 

    //One of the key features of OpenCL is its portability. So, for instance, there might be situations
    // in which both the CPU and the GPU can run OpenCL code. Thus, 
    // a good practice is to verify the OpenCL platforms to choose on which the compiled code run.

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

    // std::cout << "Double FP = " << default_device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";

    //create context, kernel source and queue to push commands to the device.
    cl::Context context({ default_device });
    cl::CommandQueue queue(context, default_device);
    cl::Program::Sources sources;

    //Appending the kernel, which is presented here as a string. 
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });


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
    cl::Buffer data_d(context, CL_MEM_READ_WRITE, sizeof(data));
    queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, sizeof(data), data);


    //OpenCL compiles the kernel in runtime, that's the reason it is expressed as a string. 
    //There are also ways to compile the device-side code offline. 
    cl::Program program(context, sources);


    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    //If runtime compilation are found they are presented in this point of the program.


    //From the program, which contains the "simple_add" kernel, create a kernel for execution
    //with three cl:buffers as parameters.
    //The types must match the arguments of the kernel function. 
    // cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
    cl::make_kernel<cl::Buffer> radix_sort8(cl::Kernel(program, "radix_sort8"));
    
    //Details to enqueue the kernel for execution.
    cl::NDRange global(NUM_SHORTS);
    // simple_add(cl::EnqueueArgs(queue, global), A_d, B_d, C_d,D_d).wait();
    radix_sort8(cl::EnqueueArgs(queue, global), data_d).wait();

    unsigned short D_h[NUM_SHORTS];
    //read result C_d from the device to array C_h
    queue.enqueueReadBuffer(data_d, CL_TRUE, 0, sizeof(data), D_h);

    std::cout << "Result: \n";
    for (int i = 0; i<NUM_SHORTS; i++) {
        std::cout << D_h[i] << " ";
    }

    std::cout << "\n";

    return 0;
}
