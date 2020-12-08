#define _USE_MATH_DEFINES

#include <algorithm>
#include <vector>

#include "mat.hpp"
#include "mod.hpp"
#include "sce.hpp"

#define NEAR_PLANE_DISTANCE 0.7f
#define FAR_PLANE_DISTANCE 50.0f
#define NEAR_PLANE_WIDTH 0.5f
#define NEAR_PLANE_HEIGHT 0.5f

float frameMatrix[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float rotatedFrameMatrix[16];
float rotationMatrix[16];
float translationMatrix[16];
float perspectiveMatrix[16];
float transformationMatrix[16];
float tmpMatrix0[16];
float tmpMatrix1[16];
float tmpMatrix2[16];

float pitch;
float yaw;
float x;
float y;
float z;

std::vector<Model *> models(0);
std::vector<float *> transformations(0);

void initializeScene() {
	pitch = 0; yaw = M_PI * 2 / 3;
	x = 3.0f, y = 3.0f, z = 3.0f;
	constructPitchMatrix(tmpMatrix0, -pitch);
	constructYawMatrix(tmpMatrix1, -yaw);
	multiplyMatrices(rotationMatrix, tmpMatrix0, tmpMatrix1);
	multiplyMatrices(tmpMatrix2, tmpMatrix1, tmpMatrix0);
	multiplyMatrices(rotatedFrameMatrix, frameMatrix, tmpMatrix2);
	constructTranslationMatrix(translationMatrix, -x, -y, -z);
	constructPerspectiveMatrix(perspectiveMatrix, NEAR_PLANE_DISTANCE, FAR_PLANE_DISTANCE, NEAR_PLANE_WIDTH, NEAR_PLANE_HEIGHT);
}

void terminateScene() {
	for (int model = models.size() - 1; model > -1; model--)
		removeModel(model);
}

void rotateCamera(float dYaw, float dPitch) {
	yaw += dYaw;
	pitch = std::max(std::min(pitch + dPitch, (float) M_PI / 2), (float) -M_PI / 2);
	constructPitchMatrix(tmpMatrix0, -pitch);
	constructYawMatrix(tmpMatrix1, -yaw);
	multiplyMatrices(rotationMatrix, tmpMatrix0, tmpMatrix1);
	multiplyMatrices(tmpMatrix2, tmpMatrix1, tmpMatrix0);
	multiplyMatrices(rotatedFrameMatrix, frameMatrix, tmpMatrix2);
}

void translateCamera(float dSway, float dHeave, float dSurge) {
	x += rotatedFrameMatrix[0] * dSway + rotatedFrameMatrix[1] * dHeave + rotatedFrameMatrix[2] * dSurge;
	y += rotatedFrameMatrix[4] * dSway + rotatedFrameMatrix[5] * dHeave + rotatedFrameMatrix[6] * dSurge;
	z += rotatedFrameMatrix[8] * dSway + rotatedFrameMatrix[9] * dHeave + rotatedFrameMatrix[10] * dSurge;
	constructTranslationMatrix(translationMatrix, -x, -y, -z);
}

void transformCamera() {
	multiplyMatrices(tmpMatrix0, perspectiveMatrix, rotationMatrix);
	multiplyMatrices(transformationMatrix, tmpMatrix0, translationMatrix);
}

void addModel(Model *model, float yaw, float pitch, float roll, float x, float y, float z) {
	models.push_back(model);
	constructRollMatrix(tmpMatrix0, roll);
	constructPitchMatrix(tmpMatrix1, pitch);
	multiplyMatrices(tmpMatrix2, tmpMatrix0, tmpMatrix1);
	constructYawMatrix(tmpMatrix0, yaw);
	multiplyMatrices(tmpMatrix1, tmpMatrix2, tmpMatrix0);
	constructTranslationMatrix(tmpMatrix0, x, y, z);
	float *modelTransformationMatrix = new float[16];
	multiplyMatrices(modelTransformationMatrix, tmpMatrix1, tmpMatrix0);
	transformations.push_back(modelTransformationMatrix);
}

void removeModel(int index) {
	delete models[index];
	delete[] transformations[index];
	models.erase(models.begin() + index);
	transformations.erase(transformations.begin() + index);
}