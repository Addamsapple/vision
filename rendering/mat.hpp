#pragma once

void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC);
void constructYawMatrix(float *matrix, float yaw);
void constructPitchMatrix(float *matrix, float pitch);
void constructRollMatrix(float *matrix, float roll);
void constructTranslationMatrix(float *matrix, float x, float y, float z);
void constructPerspectiveMatrix(float *matrix, float nearPlaneDistance, float farPlaneDistance, float nearPlaneWidth, float nearPlaneHeight);