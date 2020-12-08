/*
Written by Justin Tulleken.

png.cpp implements the PNG class defined in png.hpp. It is used to read/write/modify PNG images. This class is essentially
	a wrapper to the PNG reference library - libpng, which does all of the heavy lifting.
*/

#include <fstream>
#include "png.hpp"

//change to use const char * strings

void read(png_structp readStruct, png_bytep data, png_size_t length) {
	((std::ifstream *) png_get_io_ptr(readStruct))->read((char *) data, length);
}

void write(png_structp writeStruct, png_bytep data, png_size_t length) {
	((std::ofstream *) png_get_io_ptr(writeStruct))->write((char *) data, length);
}

void flush(png_structp writeStruct) {
	((std::ofstream *) png_get_io_ptr(writeStruct))->flush();
}

PNG::PNG(int width, int height, int type) {
	this->width = width;
	this->height = height;
	this->type = type;
	switch (type) {
		case PNG_COLOR_TYPE_GRAY: channels = 1; break;
		case PNG_COLOR_TYPE_RGB: channels = 3; break;
	}
	data = new unsigned char[width * height * channels];
}

PNG::PNG(std::string source) {
	png_structp readStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop infoStruct = png_create_info_struct(readStruct);
	std::ifstream file(source, std::ifstream::in | std::ifstream::binary);
	png_set_read_fn(readStruct, (png_voidp) &file, read);
	png_read_info(readStruct, infoStruct);
	width = png_get_image_width(readStruct, infoStruct);
	height = png_get_image_height(readStruct, infoStruct);
	channels = png_get_channels(readStruct, infoStruct);
	type = png_get_color_type(readStruct, infoStruct);
	data = new unsigned char[width * height * channels];
	for (int row = 0; row < height; row++)
		png_read_row(readStruct, data + row * width * channels, NULL);
	png_destroy_read_struct(&readStruct, &infoStruct, (png_infopp) 0);
}

PNG::~PNG() {
	delete[] data;
}

PNG * PNG::transpose() {
	PNG *result = new PNG(height, width, type);
	for (int row = 0; row < height; row++)
		for (int column = 0; column < width; column++)
			for (int channel = 0; channel < channels; channel++)
				result->data[channel + row * channels + column * height * channels] = data[channel + column * channels + row * width * channels];
	return result;
}

void PNG::reload(std::string source) {
	png_structp readStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop infoStruct = png_create_info_struct(readStruct);
	std::ifstream file(source, std::ifstream::in | std::ifstream::binary);
	png_set_read_fn(readStruct, (png_voidp) &file, read);
	for (int row = 0; row < height; row++)
		png_read_row(readStruct, data + row * width * channels, NULL);
	png_destroy_read_struct(&readStruct, &infoStruct, (png_infopp) 0);
}

void PNG::save(std::string destination) {
	png_structp writeStruct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop infoStruct = png_create_info_struct(writeStruct);
	png_set_IHDR(writeStruct, infoStruct, width, height, 8, type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	std::ofstream file(destination, std::ofstream::out | std::ofstream::binary);
	png_set_write_fn(writeStruct, (png_voidp) &file, write, flush);
	png_write_info(writeStruct, infoStruct);
	for (int row = 0; row < height; row++)
		png_write_row(writeStruct, data + row * width * channels);
	png_write_end(writeStruct, NULL);
	png_destroy_write_struct(&writeStruct, &infoStruct);
}