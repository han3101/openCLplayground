#pragma once
#include <stdint.h>
#include <cstdio>
#include <complex>

//legacy feature of C
#undef __STRICT_ANSI__
#define _USE_MATH_DEFINES 
#include <cmath>
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif

enum ImageType {
	PNG, JPG, BMP, JPEG
};


struct Image {
	uint8_t* data = NULL;
	size_t size = 0;
	int w;
	int h;
	int channels;

	Image(const char* filename, int channel_force = 0);
	Image(int w, int h, int channels = 3);
	Image(const Image& img);
	~Image();

	bool write(const char* filename);

private:
	bool read(const char* filename, int channel_force = 0);
	

public:
	ImageType get_file_type(const char* filename);

	Image& grayscale_avg_cpu();
	Image& grayscale_lum_cpu();

	Image& diffmap_cpu(Image& img);
	Image& diffmap_scale_cpu(Image& img, uint8_t scl = 0);

	Image& flipX_cpu();
	Image& flipY_cpu();

};
