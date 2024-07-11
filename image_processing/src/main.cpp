#include "image.h"
#include "opencl_image.h"
#include <cstdlib>
#include <iostream>
#include <chrono>


int main(int argc, char** argv) {
	// Image test("imgs/test.png");
    // Image testHD("imgs/testHD.jpeg");
    Image testB("imgs/testB.jpeg");
    Image testA("imgs/testA.jpeg");


    // Timing the computation
    // auto start = std::chrono::high_resolution_clock::now();

    // testA.diffmap_cpu(testB);

    // testA.write("output/diffmap_cpu.jpeg");


    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;
    
    // // High res
    // start = std::chrono::high_resolution_clock::now();

    // testHD.grayscale_avg_cpu();
    // testHD.write("imgs/grayscaleHD.jpg");

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    // // Super high res
    // start = std::chrono::high_resolution_clock::now();

    // testB.grayscale_avg_cpu();
    // testB.write("imgs/grayscaleB.jpg");

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    OpenCLImageProcessor processor;
    processor.diffmap(testA, testB);

    testA.write("output/diffmapA.jpeg");

	return 0;
}