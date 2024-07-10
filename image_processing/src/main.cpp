#include "image.h"
#include <cstdlib>
#include <iostream>
#include <chrono>


int main(int argc, char** argv) {
	Image test("imgs/test.png");

    std::cout<<test.size<<"\n";

    Image copy = test; 

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    copy.grayscale_avg_cpu();
    copy.write("imgs/grayscale.png");



    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

	return 0;
}