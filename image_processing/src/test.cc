#include <gtest/gtest.h>

#include "image.h"
#include "opencl_image.h"
#include "masks.h"
#include <cstdlib>
#include <iostream>

int is_image_black(const Image& img) {
    int isBlack = 1;

    #pragma omp parallel for num_threads(4) shared(isBlack)
    for (int i = 0; i < img.size; ++i) {
        if (img.data[i] != 0) {
            #pragma omp critical
            {
                isBlack = 0;
            }
        }
    }

    return isBlack;
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(ImageTest, BasicDiffTest) {

    Image testHD("imgs/testHD.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load image.";


    OpenCLImageProcessor processor;

    processor.diffmap(testHD, testHD);

    int is_black = is_image_black(testHD);

    ASSERT_NE(testHD.data, nullptr) << "Image dissapeared";


    EXPECT_EQ(is_black, 1) << "Diff image is not all black";
}