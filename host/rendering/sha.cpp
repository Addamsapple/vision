#include <fstream>
#include <glad.h>

#include "sha.hpp"

const int VERTEX_SHADER = 0;
const int FRAGMENT_SHADER = 1;

#define BUFFER_SIZE 1000

const int ERRORS[2] = {1000, 1001};

//needs error checking for actual shader compilation

Shader::Shader(const char *source, int type) {
	std::ifstream file(source, std::ifstream::in);
	if (file.fail())
		exit(ERRORS[0]);
	char string[BUFFER_SIZE];
	file.get(string, BUFFER_SIZE, 0);
	file.close();
	if (!file.eof())
		exit(ERRORS[1]);
	switch (type) {
		case VERTEX_SHADER: id = glCreateShader(GL_VERTEX_SHADER); break;
		case FRAGMENT_SHADER: id = glCreateShader(GL_FRAGMENT_SHADER); break;
	}
	char *pointer = &string[0];
	glShaderSource(id, 1, &pointer, NULL);
	glCompileShader(id);
}

Shader::~Shader() {
	glDeleteShader(id);
}