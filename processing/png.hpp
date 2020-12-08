/*
Written by Justin Tulleken.

png.hpp declares the PNG class implemented in png.cpp, where more information can be found.
*/

#include <png.h>
#include <string>

class PNG {
	public:
		unsigned char *data;
		int width;
		int height;
		int channels;
		int type;
		PNG(int width, int height, int type);
		PNG(std::string source);
		~PNG();
		PNG * transpose();
		void reload(std::string source);
		void save(std::string destination);
};