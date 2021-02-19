#include <algorithm>

#include <cmath>
#include <iostream>

#include <iomanip>

#include "cam.hpp"

//todo: check performance of doubles for polynomial roots

//solution sometimes not found, even for valid inputs, because roots of quartic polynomial do not converge within the number of iterations used (20 and 10)

#define REFINEMENT_ITERATIONS 20
#define FACTORIZATION_ITERATIONS 20

#define POLYNOMIAL_DEGREE 4

const float POSE_EPSILON = 0.01f;

//reorder method definitions

void pose(float *outputR, float *outputT, float *matrixI, float *matrixW);

void computePolynomialRoots(float *roots, float *polynomial);
void refinePolynomialRoots(float *roots, float *polynomial);
void trilaterateCameraPosition(float *T, float *X, float *D, int configurations);
void computeQRFactorization(float *Q, float *R, float *A);

void computeBestSolution(int *bestSolution, float *matrixU, float *matrixR, float *matrixW, float *vectorT, int solutions);

void constructPolynomial(float *polynomialCoefficients, float *tetrahedronAngles, float *tetrahedronSideLengthRatios);

void computeFeatureRadii(float *featureRayLengths, int *configurations, float *polynomialRoots, float *polynomialCoefficients, float *tetrahedronAngles, float *tetrahedronSideLengths, float *tetrahedronSideLengthRatios);
void computeCameraRotation(float *rotationMatrices, float *cameraPositions, float *featureWorldPositions, float *featureCameraPositions, int solutions);

void unprojectPixel(float *ray, float *pixel);

void multiplyMatrices(float *A, float *B, float *C, int size);
void invertMatrix(float *A, int size);
void transposeMatrix(float *A, int size);
void crossMultiplyVectors(float *A, float *B, float *C);
void normalizeVector(float *A, int size);
void combineVectors(float *A, float *B, float *C, float b, float c, int size);
void scaleVector(float *A, float *B, float b, int size);

float matrixInnerProduct(float *A, float *B, int size);
float vectorDotProduct(float *A, float *B, int size);
float vectorDistance(float *A, float *B, int size);
float vectorMagnitude(float *A, int size);

#define Cab cosines[0]
#define Cac cosines[1]
#define Cbc cosines[2]

#define Rab distances[0]
#define Rac distances[1]
#define Rbc distances[2]

#define K1 ratios[0]
#define K2 ratios[1]

void unprojectPixel(float *ray, float *point) {
	ray[0] = point[0];//(point[0] - HORIZONTAL_PRINCIPAL_POINT) / HORIZONTAL_FOCAL_LENGTH
	ray[1] = point[1];//(point[1] - VERTICAL_PRINCIPAL_POINT) / VERTICAL_FOCAL_LENGTH
	ray[2] = 1.0f;
	normalizeVector(ray, 3);
}

void projectPoint(float *image, float *world, float *vectorT, float *matrixR) {
	float translatedVector[3];
	combineVectors(translatedVector, world, vectorT, 1.0f, -1.0f, 3);
	float z = matrixInnerProduct(translatedVector, matrixR + 2, 3);
	image[0] = matrixInnerProduct(translatedVector, matrixR, 3) / z;
	image[1] = matrixInnerProduct(translatedVector, matrixR + 1, 3) / z;
}

void computeReprojectionError(float *reprojectionError, float *matrixR, float *world, float *image, float *vectorT) {
	float reprojectedPixel[2];
	projectPoint(reprojectedPixel, world, vectorT, matrixR);
	*reprojectionError = (reprojectedPixel[0] - image[0]) * (reprojectedPixel[0] - image[0]) + (reprojectedPixel[1] - image[1]) * (reprojectedPixel[1] - image[1]);
	std::cout << std::setw(15) << reprojectedPixel[0] << std::setw(15) << reprojectedPixel[1] << std::setw(15) << *reprojectionError << "\n";
}

#define RANSAC_ALGORITHM_EXECUTIONS 10

void performRandomSampleConsensus(int numPoints, float *matI, float *matW) {
	std::srand(0);
	float matrixI[4 * 3];
	float matrixW[4 * 3];
	float matrixR[3 * 3];
	float vectorT[3];
	for (int i = 0; i < RANSAC_ALGORITHM_EXECUTIONS; i++) {
		int points[4];
		for (int point = 0; point < 4; point++) {
			randomize: points[point] = (int) ((float) rand() * numPoints / (RAND_MAX + 1.0f));
			for (int priorPoint = 0; priorPoint < point; priorPoint++)
				if (points[point] == points[priorPoint])
					goto randomize;
		}
		for (int point = 0; point < 4; point++) {
			memcpy(matrixI + (long long) point * 3, matI + (long long) points[point] * 3, sizeof(float) * 3);
			memcpy(matrixW + (long long) point * 3, matW + (long long) points[point] * 3, sizeof(float) * 3);
		}
		std::cout << "iteration " << i << "\n";
		pose(matrixR, vectorT, matrixI, matrixW);
		std::cout << "\n\n\n";
	}
}

void pose(float *outputR, float *outputT, float *matrixI, float *matrixW) {
	float unprojectedPixels[3 * 3];
	for (int point = 0; point < 3; point++)
		unprojectPixel(unprojectedPixels + (long long) point * 3, matrixI + (long long) point * 3);
	float cosines[3] = {vectorDotProduct(unprojectedPixels, unprojectedPixels + 3, 3), vectorDotProduct(unprojectedPixels, unprojectedPixels + 6, 3), vectorDotProduct(unprojectedPixels + 3, unprojectedPixels + 6, 3)};
	float distances[3] = {vectorDistance(matrixW, matrixW + 3, 3), vectorDistance(matrixW, matrixW + 6, 3), vectorDistance(matrixW + 3, matrixW + 6, 3)};
	float ratios[2] = {Rbc * Rbc / (Rac * Rac), Rbc * Rbc / (Rab * Rab)};
	float polynomialCoefficients[POLYNOMIAL_DEGREE + 1];
	constructPolynomial(polynomialCoefficients, cosines, ratios);
	float roots[POLYNOMIAL_DEGREE];
	computePolynomialRoots(roots, polynomialCoefficients);
	refinePolynomialRoots(roots, polynomialCoefficients);
	float featureRadii[POLYNOMIAL_DEGREE * 2 * 3];
	int configurations;
	computeFeatureRadii(featureRadii, &configurations, roots, polynomialCoefficients, cosines, distances, ratios);
	float vectorT[POLYNOMIAL_DEGREE * 2 * 3];
	trilaterateCameraPosition(vectorT, matrixW, featureRadii, configurations);
	float matrixR[POLYNOMIAL_DEGREE * 2 * 3 * 3];
	computeCameraRotation(matrixR, vectorT, unprojectedPixels, matrixW, configurations * 2);
	int bestSolution;
	computeBestSolution(&bestSolution, matrixI, matrixR, matrixW, vectorT, configurations * 2);
	memcpy(outputR, matrixR + (long long) bestSolution * 3 * 3, sizeof(float) * 3 * 3);
	memcpy(outputT, vectorT + (long long) bestSolution * 3, sizeof(float) * 3);
}

void computeBestSolution(int *bestSolution, float *matrixU, float *matrixR, float *matrixW, float *vectorT, int solutions) {
	float translatedVector[3];
	float minimumError = FLT_MAX;
	*bestSolution = 0;
	for (int solution = 0; solution < solutions; solution++) {
		combineVectors(translatedVector, matrixW + 3 * 3, vectorT + (long long) solution * 3, 1.0f, -1.0f, 3);
		float reprojectionError;
		computeReprojectionError(&reprojectionError, matrixR + (long long) solution * 3 * 3, matrixW + 3 * 3, matrixU + 3 * 3, vectorT + (long long) solution * 3);
		if (reprojectionError < minimumError) {
			minimumError = reprojectionError;
			*bestSolution = solution;
		}
	}
}

void computeFeatureRadii(float *featureRadii, int *configurations, float *polynomialRoots, float *polynomialCoefficients, float *cosines, float *distances, float *ratios) {
	*configurations = 0;
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++) {
		float rootPowers[POLYNOMIAL_DEGREE + 1];
		rootPowers[0] = 1.0f;
		float function = polynomialCoefficients[0];
		for (int power = 1; power < POLYNOMIAL_DEGREE + 1; power++) {
			rootPowers[power] = polynomialRoots[root] * rootPowers[power - 1];
			function += polynomialCoefficients[power] * rootPowers[power];
		}
		if (std::abs(function) < POSE_EPSILON) {
			if (root > 0 && std::abs(polynomialRoots[root - 1] - rootPowers[1]) < POSE_EPSILON)
				continue;
			float a = Rab / std::sqrt(rootPowers[2] - 2.0f * rootPowers[1] * Cab + 1.0f);
			float b = a * rootPowers[1];
			float m = 1.0f - K1;
			float mp = 1.0f;
			float q = rootPowers[2] - K1;
			float qp = (1.0f - K2) * rootPowers[2] + 2.0f * K2 * Cab * rootPowers[1] - K2;
			if (std::abs(m * qp - mp * q) > POSE_EPSILON) {
				float p = 2.0f * (K1 * Cac - Cbc * rootPowers[1]);
				float pp = -2.0f * Cbc * rootPowers[1];
				float y = (pp * q - p * qp) / (m * qp - mp * q);
				float c = a * y;
				featureRadii[*configurations * 3] = a;
				featureRadii[1 + *configurations * 3] = b;
				featureRadii[2 + *configurations * 3] = c;
				(*configurations)++;
			} else {
				float y[2] = {Cac - std::sqrt(Cac * Cac + Rac * Rac / (a * a) - 1.0f), 0.0f};
				y[1] = 2.0f * Cac - y[0];
				float c[2] = {a * y[0], a * y[1]};
				for (int configuration = 0; configuration < 2; configuration++)
					if (std::abs(Rbc * Rbc - b * b - c[configuration] * c[configuration] + 2.0f * b * c[configuration] * Cbc) < POSE_EPSILON) {
						featureRadii[*configurations * 3] = a;
						featureRadii[1 + *configurations * 3] = b;
						featureRadii[2 + *configurations * 3] = c[configuration];
						(*configurations)++;
					}
			}
		}
	}
}

void constructPolynomial(float *polynomialCoefficients, float *cosines, float *ratios) {
	float leadingPolynomialCoefficient = (K1 * K2 - K1 - K2) * (K1 * K2 - K1 - K2) - 4.0f * K1 * K2 * Cbc * Cbc;
	polynomialCoefficients[0] = ((K1 * K2 + K1 - K2) * (K1 * K2 + K1 - K2) - 4.0f * K1 * K1 * K2 * Cac * Cac) / leadingPolynomialCoefficient;
	polynomialCoefficients[1] = (4.0f * (K1 * K2 + K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2.0f * K1 * K2 * Cab * Cac * Cac)) / leadingPolynomialCoefficient;
	polynomialCoefficients[2] = ((2.0f * K2 * (1.0f - K1) * Cab) * (2.0f * K2 * (1.0f - K1) * Cab) + 2.0f * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4.0f * K1 * ((K1 - K2) * Cbc * Cbc + K1 * (1.0f - K2) * Cac * Cac - 2.0f * (1.0f + K1) * K2 * Cab * Cac * Cbc)) / leadingPolynomialCoefficient;
	polynomialCoefficients[3] = (4.0f * (K1 * K2 - K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2.0f * K2 * Cab * Cbc)) / leadingPolynomialCoefficient;
	polynomialCoefficients[4] = 1.0f;
	std::cout << "x^4+" << polynomialCoefficients[3] << "x^3+" << polynomialCoefficients[2] << "x^2+" << polynomialCoefficients[1] << "x+" << polynomialCoefficients[0] << "\n";
}

void computeCameraRotation(float *matrixR, float *vectorT, float *matrixI, float *matrixW, int solutions) {
	float translatedFeatureWorldPositions[3 * 3];
	float scaledFeatureCameraPositions[3 * 3];
	for (int solution = 0; solution < solutions; solution++) {
		combineVectors(translatedFeatureWorldPositions, matrixW, vectorT + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedFeatureWorldPositions + 3, matrixW + 3, vectorT + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedFeatureWorldPositions + 2 * 3, matrixW + 2 * 3, vectorT + (long long) solution * 3, 1.0f, -1.0f, 3);
		scaleVector(scaledFeatureCameraPositions, matrixI, vectorMagnitude(translatedFeatureWorldPositions, 3), 3);
		scaleVector(scaledFeatureCameraPositions + 3, matrixI + 3, vectorMagnitude(translatedFeatureWorldPositions + 3, 3), 3);
		scaleVector(scaledFeatureCameraPositions + 2 * 3, matrixI + 2 * 3, vectorMagnitude(translatedFeatureWorldPositions + 2 * 3, 3), 3);
		invertMatrix(translatedFeatureWorldPositions, 3);
		multiplyMatrices(matrixR + (long long) solution * 3 * 3, translatedFeatureWorldPositions, scaledFeatureCameraPositions, 3);
	}
}

void computePolynomialRoots(float *polynomialRoots, float *polynomialCoefficients) {
	float matrixA[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixQ[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixR[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	memset(matrixA, 0, POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * sizeof(float));
	for (int row = 0; row < POLYNOMIAL_DEGREE - 1; row++)
		matrixA[row + 1 + row * POLYNOMIAL_DEGREE] = 1;
	for (int column = 0; column < POLYNOMIAL_DEGREE; column++)
		matrixA[column + (POLYNOMIAL_DEGREE - 1) * POLYNOMIAL_DEGREE] = -polynomialCoefficients[column];
	for (int iteration = 0; iteration < FACTORIZATION_ITERATIONS; iteration++) {
		computeQRFactorization(matrixQ, matrixR, matrixA);
		multiplyMatrices(matrixA, matrixR, matrixQ, POLYNOMIAL_DEGREE);
	}
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++)
		polynomialRoots[root] = matrixA[root + root * POLYNOMIAL_DEGREE];//I should probably sort these so I can skip duplicates
}

void computeQRFactorization(float *matrixQ, float *matrixR, float *matrixA) {
	for (int qAxis = 0; qAxis < POLYNOMIAL_DEGREE; qAxis++) {
		for (int element = 0; element < POLYNOMIAL_DEGREE; element++)
			matrixQ[element + qAxis * POLYNOMIAL_DEGREE] = matrixA[qAxis + element * POLYNOMIAL_DEGREE];
		for (int aAxis = 0; aAxis < qAxis; aAxis++)
			combineVectors(matrixQ + (long long) qAxis * POLYNOMIAL_DEGREE, matrixQ + (long long) qAxis * POLYNOMIAL_DEGREE, matrixQ + (long long) aAxis * POLYNOMIAL_DEGREE, 1.0f, -matrixInnerProduct(matrixQ + (long long) aAxis * POLYNOMIAL_DEGREE, matrixA + qAxis, POLYNOMIAL_DEGREE), POLYNOMIAL_DEGREE);
		normalizeVector(matrixQ + (long long) qAxis * POLYNOMIAL_DEGREE, POLYNOMIAL_DEGREE);
	}
	multiplyMatrices(matrixR, matrixQ, matrixA, POLYNOMIAL_DEGREE);
	transposeMatrix(matrixQ, POLYNOMIAL_DEGREE);
}

void refinePolynomialRoots(float *polynomialRoots, float *polynomialCoefficients) {
	float rootPowers[POLYNOMIAL_DEGREE + 1];
	rootPowers[0] = 1.0f;
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++)
		for (int iteration = 0; iteration < REFINEMENT_ITERATIONS; iteration++) {
			for (int power = 1; power < POLYNOMIAL_DEGREE + 1; power++)
				rootPowers[power] = polynomialRoots[root] * rootPowers[power - 1];
			float function = 0.0f;
			float firstFunctionDerivative = 0.0f;
			float secondFunctionDerivative = 0.0f;
			for (int term = 0; term < POLYNOMIAL_DEGREE + 1; term++)
				function += polynomialCoefficients[term] * rootPowers[term];
			for (int term = 1; term < POLYNOMIAL_DEGREE + 1; term++)
				firstFunctionDerivative += term * polynomialCoefficients[term] * rootPowers[term - 1];
			for (int term = 2; term < POLYNOMIAL_DEGREE + 1; term++)
				secondFunctionDerivative += term * (term - 1) * polynomialCoefficients[term] * rootPowers[term - 2];
			polynomialRoots[root] -= function * firstFunctionDerivative / (firstFunctionDerivative * firstFunctionDerivative - function * secondFunctionDerivative * 0.5f);
		}
	std::sort(polynomialRoots, polynomialRoots + POLYNOMIAL_DEGREE);
	std::cout << "roots: " << polynomialRoots[0] << " " << polynomialRoots[1] << " " << polynomialRoots[2] << " " << polynomialRoots[3] << "\n";
}

void trilaterateCameraPosition(float *vectorT, float *matrixW, float *featureRadii, int configurations) {
	float vectorB[3];
	combineVectors(vectorB, matrixW + 3, matrixW, 1.0f, -1.0f, 3);
	float vectorC[3];
	combineVectors(vectorC, matrixW + 2 * 3, matrixW, 1.0f, -1.0f, 3);
	float frameMatrix[3 * 3];
	scaleVector(frameMatrix, vectorB, 1.0f, 3);
	normalizeVector(frameMatrix, 3);
	combineVectors(frameMatrix + 3, vectorC, frameMatrix, 1.0f, -vectorDotProduct(vectorC, frameMatrix, 3), 3);
	normalizeVector(frameMatrix + 3, 3);
	crossMultiplyVectors(frameMatrix + 2 * 3, frameMatrix, frameMatrix + 3);
	float vectorBXComponent = vectorMagnitude(vectorB, 3);
	float vectorCXComponent = vectorDotProduct(vectorC, frameMatrix, 3);
	float vectorCYComponent = vectorDotProduct(vectorC, frameMatrix + 3, 3);
	for (int configuration = 0; configuration < configurations; configuration++) {
		float camereXCoordinate = (featureRadii[configuration * 3] * featureRadii[configuration * 3] - featureRadii[1 + configuration * 3] * featureRadii[1 + configuration * 3] + vectorBXComponent * vectorBXComponent) / (2.0f * vectorBXComponent);
		float cameraYCoordinate = (featureRadii[configuration * 3] * featureRadii[configuration * 3] - featureRadii[2 + configuration * 3] * featureRadii[2 + configuration * 3] + vectorCXComponent * vectorCXComponent + vectorCYComponent * vectorCYComponent - 2.0f * vectorCXComponent * camereXCoordinate) / (2.0f * vectorCYComponent);
		float cameraZCoordinate = std::sqrt(featureRadii[configuration * 3] * featureRadii[configuration * 3] - camereXCoordinate * camereXCoordinate - cameraYCoordinate * cameraYCoordinate);
		combineVectors(vectorT + (long long) configuration * 2 * 3, matrixW, frameMatrix, 1.0f, camereXCoordinate, 3);
		combineVectors(vectorT + (long long) configuration * 2 * 3, vectorT + (long long) configuration * 2 * 3, frameMatrix + 3, 1.0f, cameraYCoordinate, 3);
		combineVectors(vectorT + (long long) configuration * 2 * 3, vectorT + (long long) configuration * 2 * 3, frameMatrix + 2 * 3, 1.0f, cameraZCoordinate, 3);
		combineVectors(vectorT + (long long) configuration * 2 * 3 + 3, matrixW, frameMatrix, 1.0f, camereXCoordinate, 3);
		combineVectors(vectorT + (long long) configuration * 2 * 3 + 3, vectorT + (long long) configuration * 2 * 3 + 3, frameMatrix + 3, 1.0f, cameraYCoordinate, 3);
		combineVectors(vectorT + (long long) configuration * 2 * 3 + 3, vectorT + (long long) configuration * 2 * 3 + 3, frameMatrix + 2 * 3, 1.0f, -cameraZCoordinate, 3);
	}
	//TESTING
	for (int i = 0; i < configurations * 2; i++)
		std::cout << std::setw(15) << vectorT[i * 3] << std::setw(15) << vectorT[i * 3 + 1] << std::setw(15) << vectorT[i * 3 + 2] << "\n";
	std::cout << "\n";
}

void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC, int size) {
	for (int row = 0; row < size; row++)
		for (int column = 0; column < size; column++)
			matrixA[column + row * size] = matrixInnerProduct(matrixB + (long long) row * size, matrixC + column, size);
}

void invertMatrix(float *matrix, int size) {
	for (int iteration = 0; iteration < size; iteration++) {
		float pivot = matrix[iteration + iteration * size];
		for (int row = 0; row < size; row++)
			matrix[iteration + row * size] = -matrix[iteration + row * size] / pivot;
		for (int row = 0; row < size; row++)
			if (row != iteration)
				for (int column = 0; column < size; column++)
					if (column != iteration)
						matrix[column + row * size] = matrix[column + row * size] + matrix[column + iteration * size] * matrix[iteration + row * size];
		for (int column = 0; column < size; column++)
			matrix[column + iteration * size] = matrix[column + iteration * size] / pivot;
		matrix[iteration + iteration * size] = 1.0f / pivot;
	}
}

void transposeMatrix(float *matrix, int size) {
	for (int row = 0; row < size; row++)
		for (int column = row + 1; column < size; column++) {
			float temp = matrix[column + row * size];
			matrix[column + row * size] = matrix[row + column * size];
			matrix[row + column * size] = temp;
		}
}

void crossMultiplyVectors(float *vectorA, float *vectorB, float *vectorC) {
	vectorA[0] = vectorB[1] * vectorC[2] - vectorB[2] * vectorC[1];
	vectorA[1] = vectorB[2] * vectorC[0] - vectorB[0] * vectorC[2];
	vectorA[2] = vectorB[0] * vectorC[1] - vectorB[1] * vectorC[0];
}

void normalizeVector(float *vector, int size) {
	float magnitude = vectorMagnitude(vector, size);
	for (int element = 0; element < size; element++)
		vector[element] /= magnitude;
}

void combineVectors(float *vectorA, float *vectorB, float *vectorC, float b, float c, int size) {
	for (int element = 0; element < size; element++)
		vectorA[element] = b * vectorB[element] + c * vectorC[element];
}

void scaleVector(float *vectorA, float *vectorB, float b, int size) {
	for (int element = 0; element < size; element++)
		vectorA[element] = b * vectorB[element];
}
	
float matrixInnerProduct(float *matrixA, float *matrixB, int size) {
	float innerProduct = 0.0f;
	for (int element = 0; element < size; element++)
		innerProduct += matrixA[element] * matrixB[element * size];
	return innerProduct;
}

float vectorDotProduct(float *vectorA, float *vectorB, int size) {
	float dotProduct = 0.0f;
	for (int element = 0; element < size; element++)
		dotProduct += vectorA[element] * vectorB[element];
	return dotProduct;
}

float vectorDistance(float *vectorA, float *vectorB, int size) {
	float distance = 0.0f;
	for (int element = 0; element < size; element++) {
		float difference = vectorA[element] - vectorB[element];
		distance += difference * difference;
	}
	return std::sqrt(distance);
}

float vectorMagnitude(float *vector, int size) {
	float magnitude = 0.0f;
	for (int element = 0; element < size; element++)
		magnitude += vector[element] * vector[element];
	return std::sqrt(magnitude);
}