#pragma once

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include "image.h"
#include <stdint.h>

class OpenCLImageProcessor {
public:
    OpenCLImageProcessor();
    ~OpenCLImageProcessor();

    void init();
    void grayscale_avg(Image& image);
    void grayscale_lum(Image& image);


private:
    cl::Context context;
    cl::Device device;
    // cl::Program program;
    cl::CommandQueue queue;

    void loadKernels();
    std::string loadKernelSource(const std::string& fileName);
    void setupMemory(Image& image, cl::Buffer& bufferImage);
};