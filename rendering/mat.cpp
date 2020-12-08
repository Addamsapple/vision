#include <cmath>
#include <cstring>

void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC) {
	memset(matrixA, 0, sizeof(float) * 16);
	for (int row = 0; row < 4; row++)
		for (int column = 0; column < 4; column++)
			for (int pair = 0; pair < 4; pair++)
				matrixA[column + row * 4] += matrixB[pair + row * 4] * matrixC[column + pair * 4];
}

void constructYawMatrix(float *matrix, float yaw) {
	memset(matrix, 0, sizeof(float) * 16);
	matrix[0] = cos(yaw);
	matrix[2] = sin(yaw);
	matrix[5] = 1.0f;
	matrix[8] = -sin(yaw);
	matrix[10] = cos(yaw);
	matrix[15] = 1.0f;
}

void constructPitchMatrix(float *matrix, float pitch) {
	memset(matrix, 0, sizeof(float) * 16);
	matrix[0] = 1.0f;
	matrix[5] = cos(pitch);
	matrix[6] = -sin(pitch);
	matrix[9] = sin(pitch);
	matrix[10] = cos(pitch);
	matrix[15] = 1.0f;
}

void constructRollMatrix(float *matrix, float roll) {
	memset(matrix, 0, sizeof(float) * 16);
	matrix[0] = cos(roll);
	matrix[1] = -sin(roll);
	matrix[4] = sin(roll);
	matrix[5] = cos(roll);
	matrix[10] = 1.0f;
	matrix[15] = 1.0f;
}

void constructTranslationMatrix(float *matrix, float x, float y, float z) {
	memset(matrix, 0, sizeof(float) * 16);
	matrix[0] = 1.0f;
	matrix[3] = x;
	matrix[5] = 1.0f;
	matrix[7] = y;
	matrix[10] = 1.0f;
	matrix[11] = z;
	matrix[15] = 1.0f;
}

void constructPerspectiveMatrix(float *matrix, float nearPlaneDistance, float farPlaneDistance, float nearPlaneWidth, float nearPlaneHeight) {
	memset(matrix, 0, sizeof(float) * 16);
	matrix[0] = 2.0f * nearPlaneDistance / nearPlaneWidth;
	matrix[5] = 2.0f * nearPlaneDistance / nearPlaneHeight;
	matrix[10] = -(nearPlaneDistance + farPlaneDistance) / (farPlaneDistance - nearPlaneDistance);
	matrix[11] = -2.0f * nearPlaneDistance * farPlaneDistance / (farPlaneDistance - nearPlaneDistance);
	matrix[14] = -1.0f;
}