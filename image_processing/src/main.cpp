#include "image.h"
#include "opencl_image.h"
#include <cstdlib>
#include <iostream>
#include <chrono>


int main(int argc, char** argv) {
	// Image test("imgs/test.png");
    Image testHD("imgs/testHD.jpeg");
    // Image testB("imgs/testB.jpeg");
    // Image testA("imgs/testA.jpeg");
    // Image cat("imgs/cat.jpeg");

    Image gpu_test = testHD;

    // std::cout<<cat.channels<<"\n";

    Mask::GaussianBlur3 gaussianBlur;
    // Mask::EdgeSobelX sobelX;
    // Mask::EdgeSobelY sobelY;

    Mask::GaussianDynamic1D gaussianBlur1(1, false);
    // Mask::GaussianDynamic1D gaussianBlur2(1, true);

    // Timing the computation
    // auto start = std::chrono::high_resolution_clock::now();

    // // cat.std_convolve_clamp_to_border_cpu(0, &gaussianBlur);
    // // cat.std_convolve_clamp_to_border_cpu(1, &gaussianBlur);
    // // cat.std_convolve_clamp_to_border_cpu(2, &gaussianBlur);

    // // testHD.resizeNN(testHD.w, testHD.h * 1.5);
    // testHD.diffmap_cpu(cat);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    // testHD.write("output/convolve.jpeg");

    
    // // High res
    // start = std::chrono::high_resolution_clock::now();

    // testHD.grayscale_lum_cpu();

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;
    // testHD.write("imgs/grayscaleHD.jpg");

    // // Super high res
    // start = std::chrono::high_resolution_clock::now();

    // testB.grayscale_avg_cpu();
    // testB.write("imgs/grayscaleB.jpg");

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    OpenCLImageProcessor processor;
    // processor.std_convolve_clamp_to_0(gpu_test, &gaussianBlur1);
    // processor.std_convolve_clamp_to_0(gpu_test, &gaussianBlur2);
    // processor.std_convolve_clamp_to_0(testHD, &gaussianBlur);
    // processor.std_convolve_clamp_to_border(gpu_test, &gaussianBlur);
    // processor.diffmap(gpu_cat, test);
    // processor.std_convolve_clamp_to_0(testHD, &sobelX);
    // processor.std_convolve_clamp_to_0(testHD, &sobelY);
    // processor.resizeBicubic(gpu_test, gpu_test.w, gpu_test.h * 1.5);
    // processor.diffmap(gpu_test, testHD);

    processor.diffmap(gpu_test, gpu_test);

    // gpu_test.write("output/diff.jpeg");

    for (int i = 0; i < gpu_test.size; ++i) {
        if (gpu_test.data[i] != 0) {
            {
                std::cout<<"pixel: "<<i<<" is not black"<<"\n";
            }
        }
    }

    std::cout<<"no more black pixels!"<<"\n";

	return 0;
}