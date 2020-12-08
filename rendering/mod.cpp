#include <fstream>

#include "mod.hpp"

const int MESH = 0;
const int POINT_CLOUD = 1;

Mesh::Mesh(const char *source) {
	type = MESH;
	std::ifstream file(source, std::ifstream::in | std::ifstream::binary);
	file.read((char *) &pointCount, sizeof(unsigned int));
	file.read((char *) &triangleCount, sizeof(unsigned int));
	points = new float[pointCount * 3];
	file.read((char *) points, pointCount * sizeof(float) * 3);
	colours = new unsigned char[pointCount * 3];
	file.read((char *) colours, pointCount * sizeof(unsigned char) * 3);
	triangles = new unsigned int[triangleCount * 3];
	file.read((char *) triangles, triangleCount * sizeof(unsigned int) * 3);
	file.close();
}

Mesh::~Mesh() {
	delete[] points;
	delete[] colours;
	delete[] triangles;
}

PointCloud::PointCloud(const char *source) {
	type = POINT_CLOUD;
	std::ifstream file(source, std::ifstream::in | std::ifstream::binary);
	file.read((char *) &pointCount, sizeof(unsigned int));
	points = new float[pointCount * 3];
	file.read((char *) points, pointCount * sizeof(float) * 3);
	colours = new unsigned char[pointCount * 3];
	file.read((char *) colours, pointCount * sizeof(unsigned char) * 3);
	file.close();
}

PointCloud::~PointCloud() {
	delete[] points;
	delete[] colours;
}