#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.hpp>
#include "image.h"
#include <stdint.h>

class OpenCLImageProcessor {
public:
    OpenCLImageProcessor();
    ~OpenCLImageProcessor();

    void init();
    void grayscale_avg(Image& image);

    void diffmap(Image& image1, Image& image2);

    void flipX(Image& image);
    void flipY(Image& image);

    void std_convolve_clamp_to_0(Image& image, const Mask::BaseMask* mask);
    void std_convolve_clamp_to_border(Image& image, const Mask::BaseMask* mask);
    void std_convolve_clamp_to_cyclic(Image& image, const Mask::BaseMask* mask);

private:
    cl::Context context;
    cl::Device device;
    // cl::Program program;
    cl::CommandQueue queue;

    void loadKernels();
    std::string loadKernelSource(const std::string& fileName);
};