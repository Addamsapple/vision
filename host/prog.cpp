#include "processing/cam.hpp"
#include "processing/dep.cuh"
#include "processing/fea.cuh"
#include "processing/png.hpp"
#include "processing/pre.cuh"
#include "processing/pos.hpp"
#include "processing/tra.hpp"

#include "capturing/jpg.hpp";
#include "capturing/nets.hpp";

#include "rendering/ren.hpp"


#include <iostream>

#include <fstream>


#include <fstream>

int main() {
	//renderScene();
	initializePreprocessing();
	initializeDisparityMapComputation();
	initializeFeatureMatching();
	initializeServer();
	initJpegMem();

	bool start = true;
	int i = 0;
	for (int i = 0; i < 5; i++) {

		acquireImage();
		initializeJPGDecompression((unsigned char *) networkBuffer, 200000);
		decompressImage(leftJPG);
		acquireImage();
		initializeJPGDecompression((unsigned char *) networkBuffer, 200000);
		decompressImage(rightJPG);
		makeGray();

		rectifyImages(leftGray, rightGray, NULL);
		transposeImages();
		integrateImages();

		computeDisparityMap();
		refineDisparityMap();

		computeInterestPoints();
		if (!start) {
			matchFrames();
			setupRANSACInputs();
			updatePose();
			std::cout << "pos: " << position[0] << " " << position[1] << " " << position[2] << "\n";
		} else
			start = false;
		constructPointCloud();

		matchViews();
		computePointsFromMatches();
		fi = 1 - fi;

	}
	renderScene();

	/*initializePreprocessing();
	PNG l("ll0.png"), re("rr0.png");
	PNG ll("ll1.png"), rr("rr1.png");
	initializeDisparityMapComputation();
	rectifyImages(l.data, re.data, NULL);
	transposeImages();

	computeDisparityMap();
	refineDisparityMap();
	constructPointCloud();

	PNG rect(RECTIFIED_IMAGE_HEIGHT, RECTIFIED_IMAGE_WIDTH, PNG_COLOR_TYPE_GRAY);
	rect.data = dm;
	rect.transpose()->save("dmap.png");

	integrateImages();
	initializeFeatureMatching();
	computeInterestPoints();
	matchViews();
	computePointsFromMatches();
	
	fi =  1 - fi;

	rectifyImages(ll.data, rr.data, NULL);
	transposeImages();
	integrateImages();
	computeInterestPoints();
	matchFrames();

	computeDisparityMap();
	refineDisparityMap();
	rect.data = dm;
	rect.transpose()->save("dmap2.png");
	//constructPointCloud();

	setupRANSACInputs();
	std::cout << "retained inputs: " << retainedFeatures << "\n";
	updatePose();
	constructPointCloud();
	std::cout << "pos: " << position[0] << " " << position[1] << " " <<  position[2];

	float or[9];
	//float tr[3];
	//performRandomSampleConsensus(or, tr, retainedPixels, retainedPoints, retainedFeatures);

	//std::cout << "orientation: \n";
	///std::cout << or[0] << " " << or[3] << " " << or[6] << "\n";
	//std::cout << or[1] << " " << or[4] << " " << or[7] << "\n";
	//std::cout << or[2] << " " << or[5] << " " << or[8] << "\n\n";
	//std::cout << "pos: " << tr[0] << " " << tr[1] << " " <<  tr[2];
	*/
	//renderScene();
	return 0;
}