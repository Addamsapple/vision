#include <glad.h>

#include "pro.hpp"

Program::Program(){
	id = glCreateProgram();
}

Program::~Program() {
	for (int index = 0; index < shaders.size(); index++)
		delete shaders[index];
	glDeleteProgram(id);
}

void Program::add(Shader *shader) {
	shaders.push_back(shader);
	glAttachShader(id, (*(shaders[shaders.size() - 1])).id);
}

void Program::compile() {
	glLinkProgram(id);
}

void Program::use() {
	glUseProgram(id);
}