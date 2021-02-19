//#include "rendering/ren.hpp"

#include "processing/cam.hpp"
#include "processing/dep.cuh"
#include "processing/fea.cuh"
#include "processing/png.hpp"
#include "processing/pre.cuh"
#include "processing/pos.hpp"

int main() {
	/*PNG l("left6.png");
	PNG r("right6.png");
	PNG o(RECTIFIED_IMAGE_HEIGHT, RECTIFIED_IMAGE_WIDTH, PNG_COLOR_TYPE_GRAY);
	PNG b(RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT, PNG_COLOR_TYPE_GRAY);
	initializePreprocessing();
	initializeDisparityMapComputation();
	rectifyImages(l.data, r.data, o.data);
	//integrateImages();
	transposeImages();
	computeDisparityMap();
	refineDisparityMap(o.data, 4);
	o.transpose()->save("map2.png");*/

	//pose(NULL, NULL);
	//float *w = new float[3 * 3]{1.5, 7, 2, 3, 7, 0, 5, 6, 0.5};
	
	//float w[4 * 3] = {1.5f, 7.0f, 2.0f, 3.0f, 7.0f, 0.0f, 5.0f, 6.0f, 0.5f, 4.0f, 9.0f, -1.0f};
	//float w[4 * 3] = {4.0f, 9.0f, -1.0f, 1.5f, 7.0f, 2.0f, 3.0f, 7.0f, 0.0f, 5.0f, 6.0f, 0.5f};
	float w[4 * 3] = {1.5f, 7.0f, 2.0f, 5.0f, 6.0f, 0.5f, 4.0f, 9.0f, -1.0f, 3.0f, 7.0f, 0.0f};
	
	//float *p = new float[3 * 3]{ 1.5, 1, 3, 0, -1, 3, -2, -0.5, 2 };//should be normalized with respect to z dimension, though not an issue at present
	//float *p = new float[3 * 3]{0.5f, 0.33333333f, 1.0f, 0, -0.3333333f, 1.0f, -1.0f, -0.25f, 1.0f};
	
	
	//float p[4 * 3] = {0.5f, 0.33333333f, 1.0f, 0, -0.3333333f, 1.0f, -1.0f, -0.25f, 1.0f, -0.2f, -0.4f, 1.0f};
	//float p[4 * 3] = {-0.2f, -0.4f, 1.0f, 0.5f, 0.33333333f, 1.0f, 0, -0.3333333f, 1.0f, -1.0f, -0.25f, 1.0f};
	float p[4 * 3] = {0.5f, 0.33333333f, 1.0f, -1.0f, -0.25f, 1.0f, -0.2f, -0.4f, 1.0f, 0, -0.3333333f, 1.0f};

	//not working for following order: a, c, d, b
	
	//float rot[9];
	//float trans[3];
	//pose(rot, trans, p, w);
	performRandomSampleConsensus(4, p, w);



	//invert(mat, 3);
	//trilaterate(NULL, NULL, NULL, 0);

	
	/*initializePreprocessing();
	PNG l("left2.png"), re("right2.png");
	cudaMemcpy(d_lri, l.data, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rri, re.data, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
	integrateImages();
	initializeFeatureMatching();
	computeInterestPoints();
	matchInterestPoints();*/
	
	//renderScene();
	return 0;
}