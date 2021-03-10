#include <jpeglib.h>
#include "../processing/cam.hpp"

#define JPG_BUFFER_SIZE 1024 * 1024 * 7

unsigned char *leftJPG;
unsigned char *rightJPG;

unsigned char *leftGray;
unsigned char *rightGray;

jpeg_decompress_struct cinfo;
jpeg_error_mgr error;

void initializeJPGDecompression(unsigned char *buffer, int size) {
	cinfo.err = jpeg_std_error(&error);
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, buffer, size);
}

void initJpegMem() {
	leftJPG = new unsigned char[JPG_BUFFER_SIZE];
	rightJPG = new unsigned char[JPG_BUFFER_SIZE];
	leftGray = new unsigned char[DISTORTED_IMAGE_WIDTH * DISTORTED_IMAGE_HEIGHT * sizeof(unsigned char)];
	rightGray = new unsigned char[DISTORTED_IMAGE_WIDTH * DISTORTED_IMAGE_HEIGHT * sizeof(unsigned char)];
}

void makeGray() {
	for (int row = 0; row < DISTORTED_IMAGE_HEIGHT; row++)
		for (int column = 0; column < DISTORTED_IMAGE_WIDTH; column++) {
			short sum = 0;
			sum += leftJPG[column * 3 + row * DISTORTED_IMAGE_WIDTH * 3];
			sum += leftJPG[column * 3 + 1 + row * DISTORTED_IMAGE_WIDTH * 3];
			sum += leftJPG[column * 3 + 2 + row * DISTORTED_IMAGE_WIDTH * 3];
			leftGray[column + row * DISTORTED_IMAGE_WIDTH] = (unsigned char) (sum / 3);
			sum = 0;
			sum += rightJPG[column * 3 + row * DISTORTED_IMAGE_WIDTH * 3];
			sum += rightJPG[column * 3 + 1 + row * DISTORTED_IMAGE_WIDTH * 3];
			sum += rightJPG[column * 3 + 2 + row * DISTORTED_IMAGE_WIDTH * 3];
			rightGray[column + row * DISTORTED_IMAGE_WIDTH] = (unsigned char) (sum / 3);
		}
}

#include <iostream>

void decompressImage(unsigned char *buffer) {
	jpeg_read_header(&cinfo, true);
	jpeg_start_decompress(&cinfo);
	unsigned char *scanline[1];
	while (cinfo.output_scanline < cinfo.output_height) {
		scanline[0] = buffer + (long long) cinfo.output_scanline * cinfo.output_width * cinfo.output_components;
		jpeg_read_scanlines(&cinfo, scanline, 1);
	}
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
}