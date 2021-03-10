#pragma once

extern const int VERTEX_SHADER;
extern const int FRAGMENT_SHADER;

class Shader {
	public:
		unsigned int id;
		Shader(const char *source, int type);
		~Shader();
};