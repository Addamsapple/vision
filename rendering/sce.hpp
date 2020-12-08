#pragma once

#include <vector>

#include "mod.hpp"

extern float transformationMatrix[16];
extern std::vector<Model *> models;
extern std::vector<float *> transformations;

void initializeScene();
void terminateScene();
void rotateCamera(float dYaw, float dPitch);
void translateCamera(float dSway, float dHeave, float dSurge);
void transformCamera();
void addModel(Model *model, float yaw, float pitch, float roll, float x, float y, float z);
void removeModel(int index);