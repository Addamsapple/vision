#pragma once

#include <vector>

#include "sha.hpp"

class Program {
	public:
		unsigned int id;
		std::vector<Shader *> shaders;
		Program();
		~Program();
		void add(Shader *shader);
		void compile();
		void use();
};