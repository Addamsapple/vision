#include <jpeglib.h>

jpeg_decompress_struct info;//originally had problems when using any name besides "cinfo"
jpeg_error_mgr error;

void initializeJPG(unsigned char *buffer, int size) {
	info.err = jpeg_std_error(&error);
	jpeg_create_decompress(&info);
	jpeg_mem_src(&info, buffer, size);
}

void decompressData(unsigned char *buffer) {
	jpeg_read_header(&info, true);
	jpeg_start_decompress(&info);
	unsigned char *scanline[1];
	while (info.output_scanline < info.output_height) {
		scanline[0] = buffer + info.output_scanline * info.output_width * info.output_components;
		jpeg_read_scanlines(&info, scanline, 1);
	}
	jpeg_finish_decompress(&info);
	//jpeg_destroy_decompress(&cinfo);
}