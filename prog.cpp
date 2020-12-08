//#include "rendering/ren.hpp"

#include "processing/pre.cuh"

#include "processing/png.hpp"

#include "processing/cam.hpp"

#include "processing/dep.cuh"

int main() {
	PNG l("left6.png");
	PNG r("right6.png");
	PNG o(RECTIFIED_IMAGE_HEIGHT, RECTIFIED_IMAGE_WIDTH, PNG_COLOR_TYPE_GRAY);
	initializePreprocessing();
	initializeDisparityMapComputation();
	rectifyImages(l.data, r.data, o.data);
	transposeRectifiedImages();
	computeDisparityMap();
	refineDisparityMap(o.data, 4);

	o.transpose()->save("map2.png");
	
	//renderScene();
	return 0;
}