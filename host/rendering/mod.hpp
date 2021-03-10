#pragma once

extern const int MESH;
extern const int POINT_CLOUD;

class Wireframe;

class Model {
	public:
		int type = -1;
		unsigned int pointCount;
		float *points;
		unsigned char *colours;
};

class Mesh: public Model {
	public:
		unsigned int triangleCount;
		unsigned int *triangles;
		Mesh(const char *source);
		~Mesh();
};

class PointCloud: public Model {
	public:
		PointCloud(const char *source);
		~PointCloud();
};