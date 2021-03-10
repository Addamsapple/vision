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
	std::ifstream p("points.p", std::ifstream::binary);
	std::ifstream c("colours.p", std::ifstream::binary);
	pointCount = 0;
	float point[3];
	while (!p.eof()) {
		p.read((char *) point, sizeof(float) * 3);
		pointCount++;
	}
	pointCount--;
	points = new float[(long long) pointCount * 3];
	colours = new unsigned char[(long long) pointCount * 3];
	p.clear();
	p.seekg(0, std::ifstream::beg);
	p.read((char *) points, pointCount * sizeof(float) * 3);
	p.close();
	c.read((char *) colours, pointCount * sizeof(unsigned char) * 3);
	c.close();
}

PointCloud::~PointCloud() {
	delete[] points;
	delete[] colours;
}