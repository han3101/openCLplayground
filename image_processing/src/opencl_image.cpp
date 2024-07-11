#include "../include/opencl_image.h"
#include <fstream>
#include <iostream>
#include<cstdlib>


OpenCLImageProcessor::OpenCLImageProcessor() {
    init();
}

OpenCLImageProcessor::~OpenCLImageProcessor() {}

void OpenCLImageProcessor::init() {
    // Select Platform
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // Select a device
    // TODO make this configurable
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    device = all_devices[0];
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    size_t max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << "Maximum work-group size: " << max_work_group_size << "\n";

    // Define properties for the command queue
#ifdef PROFILE
    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
#else
    cl_command_queue_properties properties[] = {0UL};
#endif

    //create context, kernel source and queue to push commands to the device.
    context = cl::Context({ device });
    queue = cl::CommandQueue(context, device, properties);

}

std::string OpenCLImageProcessor::loadKernelSource(const std::string& fileName) {
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

void OpenCLImageProcessor::grayscale_avg(Image& image) {

    if(image.channels < 3) {
		std::cout<<"Image "<<&image<<" has less than 3 channels, it is assumed to already be grayscale."<<std::endl;
        return;
	}

    // Prepare memory
    int bytes_i = image.size * sizeof(uint8_t);
    cl::Buffer data_d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/grayscale.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "grayscale_avg");
    kernel.setArg(0, data_d);
    kernel.setArg(1, image.channels);

    // Set dimensions
    cl::NDRange global(image.w * image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    queue.finish();

    // Read back the results
    queue.enqueueReadBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::diffmap(Image& image1, Image& image2) {

    // Prepare memory
    int bytes_i = image1.size * sizeof(uint8_t);
    int bytes_o = image2.size * sizeof(uint8_t);
    cl::Buffer image1_d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes_i, image1.data);
    cl::Buffer image2_d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_o, image2.data);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/diffmap.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }

    // Preprocessing
    int compare_width = fmin(image1.w,image2.w);
	int compare_height = fmin(image1.h,image2.h);
	int compare_channels = fmin(image1.channels,image2.channels);

    // Load in kernel args
    cl::Kernel kernel(program, "diffmap");
    kernel.setArg(0, image1_d);
    kernel.setArg(1, image2_d);
    kernel.setArg(2, image1.w);
    kernel.setArg(3, image1.h);
    kernel.setArg(4, image1.channels);
    kernel.setArg(5, image2.w);
    kernel.setArg(6, image2.h);
    kernel.setArg(7, image2.channels);
    kernel.setArg(8, compare_width);
    kernel.setArg(9, compare_height);
    kernel.setArg(10,compare_channels);

    // Set dimensions
    cl::NDRange global(image1.w, image1.h, image1.channels);

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    queue.finish();

    // Read back the results
    queue.enqueueReadBuffer(image1_d, CL_TRUE, 0, bytes_i, image1.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}