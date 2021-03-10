#include "fea.cuh"

extern float orientation[9];

extern float position[3];

extern float points[MAXIMUM_INTEREST_POINTS * 3];

extern int retainedFeatures;

extern float retainedPoints[MAXIMUM_INTEREST_POINTS * 3];
extern float retainedPixels[MAXIMUM_INTEREST_POINTS * 3];

void setupRANSACInputs();
void computePointsFromMatches();
void updatePose();