#include <algorithm>

#include "cam.hpp"
#include "fea.cuh"
#include "pos.hpp"

#define VERTICAL_MATCHING_THRESHOLD 18.0f

float orientation[3 * 3] = {
	1.0f,	0.0f,	0.0f,
	0.0f,	1.0f,	0.0f,
	0.0f,	0.0f,	1.0f
};

float position[3] = {0.0f, 0.0f, 0.0f};

float points[MAXIMUM_INTEREST_POINTS * 3];

int retainedFeatures;

float retainedPoints[MAXIMUM_INTEREST_POINTS * 3];
float retainedPixels[MAXIMUM_INTEREST_POINTS * 3];

#include <iostream>

void computePointsFromMatches() {
	float point[3];
	for (int match = 0; match < ipmc; match++) {
		float pixelRow = (lrip[ipm[match]] + rrip[ipm[match + MAXIMUM_INTEREST_POINTS]]) * 0.5f;
		float pixelColumn = lrip[ipm[match] + MAXIMUM_INTEREST_POINTS];
		float pixelDisparity = pixelColumn - rrip[ipm[match + MAXIMUM_INTEREST_POINTS] + MAXIMUM_INTEREST_POINTS];
		if (std::abs(pixelRow - rrip[ipm[match + MAXIMUM_INTEREST_POINTS]]) < VERTICAL_MATCHING_THRESHOLD * 0.5f && pixelDisparity > 0.0f) {
			point[0] = (pixelColumn + LEFT_RECTIFIED_COLUMN - HORIZONTAL_PRINCIPAL_POINT) * BASELINE / pixelDisparity;
			point[1] = (pixelRow + TOP_RECTIFIED_ROW - VERTICAL_PRINCIPAL_POINT) * BASELINE / pixelDisparity;
			point[2] = HORIZONTAL_FOCAL_LENGTH * BASELINE / pixelDisparity;
			//std::cout << "disp: " << pixelDisparity << "depth: " << point[2] << "\n";
			for (int dimension = 0; dimension < 3; dimension++)
				points[dimension + ipm[match] * 3] = vectorDotProduct(point, orientation + (long long) dimension * 3, 3);//matrixInnerProduct(point, orientation + dimension, 3);
			combineVectors(points + (long long) ipm[match] * 3, points + (long long) ipm[match] * 3, position, 1.0f, 1.0f, 3);
		} else
			points[2 + ipm[match] * 3] = 0.0f;
	}
}

void setupRANSACInputs() {
	retainedFeatures = 0;
	for (int match = 0; match < ipmc; match++)
		if (points[2 + ipm[match] * 3] != 0.0f) {
			memcpy(retainedPoints + (long long) retainedFeatures * 3, points + (long long) ipm[match] * 3, sizeof(float) * 3);
			retainedPixels[retainedFeatures * 3] = lrip[ipm[match + MAXIMUM_INTEREST_POINTS] + MAXIMUM_INTEREST_POINTS];
			retainedPixels[1 + retainedFeatures * 3] = lrip[ipm[match + MAXIMUM_INTEREST_POINTS]];
			retainedPixels[2 + retainedFeatures * 3] = 1.0f;
			retainedFeatures++;
			points[2 + ipm[match] * 3] = 0.0f;
		}
}

void updatePose() {
	performRandomSampleConsensus(orientation, position, retainedPixels, retainedPoints, retainedFeatures);
}