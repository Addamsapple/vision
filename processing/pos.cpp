#include <algorithm>
//#include <cmath> - might be needed for memcpy

#include <iostream>
#include <iomanip>

#include "cam.hpp"

#define REFINEMENT_ITERATIONS 200
#define FACTORIZATION_ITERATIONS 200

#define POLYNOMIAL_DEGREE 4

#define POSE_EPSILON 0.01f

void selectUniqueFeatures(float *sampledPixels, float *sampledPoints, float *pixels, float *points, int features);

void computeCameraPose(float *rotation, float *position, float *pixels, float *points);
void constructQuarticPolynomial(float *coefficients, float *cosines, float *ratios);
void computePolynomialRoots(float *roots, float *coefficients);
void computeQRFactorization(float *Q, float *R, float *A);
void refinePolynomialRoots(float *roots, float *coefficients);
void computeConfigurations(float *rayLengths, int *configurations, float *roots, float *coefficients, float *cosines, float *distances, float *ratios);
void trilaterateCameraPosition(float *positions, float *points, float *radii, int configurations);
void computeCameraOrientation(float *orientations, float *positions, float *rays, float *points, int solutions);
void selectBestSolution(float *bestOrientation, float *bestPosition, float *orientations, float *positions, float *pixels, float *points, int solutions);

void computeReprojectionError(float *reprojectionError, float *orientation, float *position, float *pixel, float *point);
void projectPoint(float *pixel, float *orientation, float *position, float *point);
void unprojectPixel(float *ray, float *pixel);

void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC, int size);
void invertMatrix(float *matrix, int size);
void transposeMatrix(float *matrix, int size);
void crossMultiplyVectors(float *vectorA, float *vectorB, float *vectorC);
void normalizeVector(float *vector, int size);
void combineVectors(float *vectorA, float *vectorB, float *vectorC, float b, float c, int size);
void scaleVector(float *vectorA, float *vectorB, float b, int size);

float matrixInnerProduct(float *matrixA, float *matrixB, int size);
float vectorDotProduct(float *vectorA, float *vectorB, int size);
float vectorDistance(float *vectorA, float *vectorB, int size);
float vectorMagnitude(float *vector, int size);

#define RANSAC_ALGORITHM_ITERATIONS 10

#define RANSAC_INLIER_ERROR_THRESHOLD 0.0001f

void performRandomSampleConsensus(float *orientation, float *position, float *pixels, float *points, int features) {
	float sampledPixels[4 * 3];
	float sampledPoints[4 * 3];
	float orientations[RANSAC_ALGORITHM_ITERATIONS * 3 * 3];
	float positions[RANSAC_ALGORITHM_ITERATIONS * 3];
	int bestIteration = 0;
	int mostInliers = 0;
	float lowestError = FLT_MAX;
	for (int iteration = 0; iteration < RANSAC_ALGORITHM_ITERATIONS; iteration++) {
		selectUniqueFeatures(sampledPixels, sampledPoints, pixels, points, features);
		computeCameraPose(orientations + (long long) iteration * 3 * 3, positions + (long long) iteration * 3, sampledPixels, sampledPoints);
		float reprojectionError;
		float totalError = 0.0f;
		int inliers = 0;
		for (int feature = 0; feature < features; feature++) {
			computeReprojectionError(&reprojectionError, orientations + (long long) iteration * 3 * 3, positions + (long long) iteration * 3, pixels + (long long) feature * 3, points + (long long) feature * 3);
			if (reprojectionError < RANSAC_INLIER_ERROR_THRESHOLD) {
				totalError += reprojectionError;
				inliers++;
			}
		}
		if (inliers > mostInliers) {
			bestIteration = iteration;
			mostInliers = inliers;
			lowestError = totalError;
		} else if (inliers == mostInliers && totalError < lowestError) {
			bestIteration = iteration;
			lowestError = totalError;
		}
	}
	memcpy(orientation, orientations + (long long) bestIteration * 3 * 3, sizeof(float) * 3 * 3);
	memcpy(position, positions + (long long) bestIteration * 3, sizeof(float) * 3);
	std::cout << "most inliers: " << mostInliers << "\n";
	std::cout << "lowest total error: " << lowestError << "\n";
}

void selectUniqueFeatures(float *sampledPixels, float *sampledPoints, float *pixels, float *points, int features) {
	int sampledFeatures[4];
	for (int feature = 0; feature < 4; feature++) {
		randomize: sampledFeatures[feature] = (int) ((float) rand() * features / (RAND_MAX + 1.0f));
		for (int priorFeature = 0; priorFeature < feature; priorFeature++)
			if (sampledFeatures[feature] == sampledFeatures[priorFeature])
				goto randomize;
	}
	for (int feature = 0; feature < 4; feature++) {
		memcpy(sampledPixels + (long long) feature * 3, pixels + (long long) sampledFeatures[feature] * 3, sizeof(float) * 3);
		memcpy(sampledPoints + (long long) feature * 3, points + (long long) sampledFeatures[feature] * 3, sizeof(float) * 3);
	}
}

#define Cab cosines[0]
#define Cac cosines[1]
#define Cbc cosines[2]

#define Rab distances[0]
#define Rac distances[1]
#define Rbc distances[2]

#define K1 ratios[0]
#define K2 ratios[1]

void computeCameraPose(float *orientation, float *position, float *pixels, float *points) {
	float unprojectedPixels[3 * 3];
	for (int point = 0; point < 3; point++)
		unprojectPixel(unprojectedPixels + (long long) point * 3, pixels + (long long) point * 3);
	float cosines[3] = {vectorDotProduct(unprojectedPixels, unprojectedPixels + 3, 3), vectorDotProduct(unprojectedPixels, unprojectedPixels + 6, 3), vectorDotProduct(unprojectedPixels + 3, unprojectedPixels + 6, 3)};
	float distances[3] = {vectorDistance(points, points + 3, 3), vectorDistance(points, points + 6, 3), vectorDistance(points + 3, points + 6, 3)};
	float ratios[2] = {Rbc * Rbc / (Rac * Rac), Rbc * Rbc / (Rab * Rab)};
	float polynomialCoefficients[POLYNOMIAL_DEGREE + 1];
	constructQuarticPolynomial(polynomialCoefficients, cosines, ratios);
	float roots[POLYNOMIAL_DEGREE];
	computePolynomialRoots(roots, polynomialCoefficients);
	refinePolynomialRoots(roots, polynomialCoefficients);
	float rayLengths[POLYNOMIAL_DEGREE * 2 * 3];
	int configurations;
	computeConfigurations(rayLengths, &configurations, roots, polynomialCoefficients, cosines, distances, ratios);
	float positions[POLYNOMIAL_DEGREE * 2 * 3];
	trilaterateCameraPosition(positions, points, rayLengths, configurations);
	float orientations[POLYNOMIAL_DEGREE * 2 * 3 * 3];
	computeCameraOrientation(orientations, positions, unprojectedPixels, points, configurations * 2);
	selectBestSolution(orientation, position, orientations, positions, pixels, points, configurations * 2);
}

void constructQuarticPolynomial(float *coefficients, float *cosines, float *ratios) {
	float leadingCoefficient = (K1 * K2 - K1 - K2) * (K1 * K2 - K1 - K2) - 4.0f * K1 * K2 * Cbc * Cbc;
	coefficients[0] = ((K1 * K2 + K1 - K2) * (K1 * K2 + K1 - K2) - 4.0f * K1 * K1 * K2 * Cac * Cac) / leadingCoefficient;
	coefficients[1] = (4.0f * (K1 * K2 + K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2.0f * K1 * K2 * Cab * Cac * Cac)) / leadingCoefficient;
	coefficients[2] = ((2.0f * K2 * (1.0f - K1) * Cab) * (2.0f * K2 * (1.0f - K1) * Cab) + 2.0f * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4.0f * K1 * ((K1 - K2) * Cbc * Cbc + K1 * (1.0f - K2) * Cac * Cac - 2.0f * (1.0f + K1) * K2 * Cab * Cac * Cbc)) / leadingCoefficient;
	coefficients[3] = (4.0f * (K1 * K2 - K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2.0f * K2 * Cab * Cbc)) / leadingCoefficient;
	coefficients[4] = 1.0f;
}

void computePolynomialRoots(float *roots, float *coefficients) {
	float matrixA[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixQ[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixR[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	memset(matrixA, 0, POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE * sizeof(float));
	for (int row = 0; row < POLYNOMIAL_DEGREE - 1; row++)
		matrixA[row + 1 + row * POLYNOMIAL_DEGREE] = 1;
	for (int column = 0; column < POLYNOMIAL_DEGREE; column++)
		matrixA[column + (POLYNOMIAL_DEGREE - 1) * POLYNOMIAL_DEGREE] = -coefficients[column];
	for (int iteration = 0; iteration < FACTORIZATION_ITERATIONS; iteration++) {
		computeQRFactorization(matrixQ, matrixR, matrixA);
		multiplyMatrices(matrixA, matrixR, matrixQ, POLYNOMIAL_DEGREE);
	}
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++)
		roots[root] = matrixA[root + root * POLYNOMIAL_DEGREE];
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

void refinePolynomialRoots(float *roots, float *coefficients) {
	float rootPowers[POLYNOMIAL_DEGREE + 1];
	rootPowers[0] = 1.0f;
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++)
		for (int iteration = 0; iteration < REFINEMENT_ITERATIONS; iteration++) {
			for (int power = 1; power < POLYNOMIAL_DEGREE + 1; power++)
				rootPowers[power] = roots[root] * rootPowers[power - 1];
			float function = 0.0f;
			float firstDerivative = 0.0f;
			float secondDerivative = 0.0f;
			for (int term = 0; term < POLYNOMIAL_DEGREE + 1; term++)
				function += coefficients[term] * rootPowers[term];
			for (int term = 1; term < POLYNOMIAL_DEGREE + 1; term++)
				firstDerivative += term * coefficients[term] * rootPowers[term - 1];
			for (int term = 2; term < POLYNOMIAL_DEGREE + 1; term++)
				secondDerivative += term * (term - 1) * coefficients[term] * rootPowers[term - 2];
			roots[root] -= function * firstDerivative / (firstDerivative * firstDerivative - function * secondDerivative * 0.5f);
		}
	std::sort(roots, roots + POLYNOMIAL_DEGREE);
}

void computeConfigurations(float *rayLengths, int *configurations, float *roots, float *coefficients, float *cosines, float *distances, float *ratios) {
	*configurations = 0;
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++) {
		float rootPowers[POLYNOMIAL_DEGREE + 1];
		rootPowers[0] = 1.0f;
		float function = coefficients[0];
		for (int power = 1; power < POLYNOMIAL_DEGREE + 1; power++) {
			rootPowers[power] = roots[root] * rootPowers[power - 1];
			function += coefficients[power] * rootPowers[power];
		}
		if (std::abs(function) < POSE_EPSILON) {
			if (root > 0 && std::abs(roots[root - 1] - rootPowers[1]) < POSE_EPSILON)
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
				rayLengths[*configurations * 3] = a;
				rayLengths[1 + *configurations * 3] = b;
				rayLengths[2 + *configurations * 3] = c;
				(*configurations)++;
			} else {
				float y[2] = {Cac - std::sqrt(Cac * Cac + Rac * Rac / (a * a) - 1.0f), 0.0f};
				y[1] = 2.0f * Cac - y[0];
				float c[2] = {a * y[0], a * y[1]};
				for (int configuration = 0; configuration < 2; configuration++)
					if (std::abs(Rbc * Rbc - b * b - c[configuration] * c[configuration] + 2.0f * b * c[configuration] * Cbc) < POSE_EPSILON) {
						rayLengths[*configurations * 3] = a;
						rayLengths[1 + *configurations * 3] = b;
						rayLengths[2 + *configurations * 3] = c[configuration];
						(*configurations)++;
					}
			}
		}
	}
}

void trilaterateCameraPosition(float *positions, float *points, float *radii, int configurations) {
	float pointB[3];
	combineVectors(pointB, points + 3, points, 1.0f, -1.0f, 3);
	float pointC[3];
	combineVectors(pointC, points + 2 * 3, points, 1.0f, -1.0f, 3);
	float frameMatrix[3 * 3];
	scaleVector(frameMatrix, pointB, 1.0f, 3);
	normalizeVector(frameMatrix, 3);
	combineVectors(frameMatrix + 3, pointC, frameMatrix, 1.0f, -vectorDotProduct(pointC, frameMatrix, 3), 3);
	normalizeVector(frameMatrix + 3, 3);
	crossMultiplyVectors(frameMatrix + 2 * 3, frameMatrix, frameMatrix + 3);
	float pointBCoordinateX = vectorMagnitude(pointB, 3);
	float pointCCoordinateX = vectorDotProduct(pointC, frameMatrix, 3);
	float pointCCoordinateY = vectorDotProduct(pointC, frameMatrix + 3, 3);
	for (int configuration = 0; configuration < configurations; configuration++) {
		float cameraCoordinateX = (radii[configuration * 3] * radii[configuration * 3] - radii[1 + configuration * 3] * radii[1 + configuration * 3] + pointBCoordinateX * pointBCoordinateX) / (2.0f * pointBCoordinateX);
		float cameraCoordinateY = (radii[configuration * 3] * radii[configuration * 3] - radii[2 + configuration * 3] * radii[2 + configuration * 3] + pointCCoordinateX * pointCCoordinateX + pointCCoordinateY * pointCCoordinateY - 2.0f * pointCCoordinateX * cameraCoordinateX) / (2.0f * pointCCoordinateY);
		float cameraCoordinateZ = std::sqrt(radii[configuration * 3] * radii[configuration * 3] - cameraCoordinateX * cameraCoordinateX - cameraCoordinateY * cameraCoordinateY);
		combineVectors(positions + (long long) configuration * 2 * 3, points, frameMatrix, 1.0f, cameraCoordinateX, 3);
		combineVectors(positions + (long long) configuration * 2 * 3, positions + (long long) configuration * 2 * 3, frameMatrix + 3, 1.0f, cameraCoordinateY, 3);
		combineVectors(positions + (long long) configuration * 2 * 3, positions + (long long) configuration * 2 * 3, frameMatrix + 2 * 3, 1.0f, cameraCoordinateZ, 3);
		combineVectors(positions + (long long) configuration * 2 * 3 + 3, points, frameMatrix, 1.0f, cameraCoordinateX, 3);
		combineVectors(positions + (long long) configuration * 2 * 3 + 3, positions + (long long) configuration * 2 * 3 + 3, frameMatrix + 3, 1.0f, cameraCoordinateY, 3);
		combineVectors(positions + (long long) configuration * 2 * 3 + 3, positions + (long long) configuration * 2 * 3 + 3, frameMatrix + 2 * 3, 1.0f, -cameraCoordinateZ, 3);
	}
}

void computeCameraOrientation(float *orientations, float *positions, float *rays, float *points, int solutions) {
	float translatedPoints[3 * 3];
	float scaledRays[3 * 3];
	for (int solution = 0; solution < solutions; solution++) {
		combineVectors(translatedPoints, points, positions + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedPoints + 3, points + 3, positions + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedPoints + 2 * 3, points + 2 * 3, positions + (long long) solution * 3, 1.0f, -1.0f, 3);
		scaleVector(scaledRays, rays, vectorMagnitude(translatedPoints, 3), 3);
		scaleVector(scaledRays + 3, rays + 3, vectorMagnitude(translatedPoints + 3, 3), 3);
		scaleVector(scaledRays + 2 * 3, rays + 2 * 3, vectorMagnitude(translatedPoints + 2 * 3, 3), 3);
		invertMatrix(translatedPoints, 3);
		multiplyMatrices(orientations + (long long) solution * 3 * 3, translatedPoints, scaledRays, 3);
	}
}

void selectBestSolution(float *bestOrientation, float *bestPosition, float *orientations, float *positions, float *pixels, float *points, int solutions) {
	float translatedPoint[3];
	float minimumError = FLT_MAX;
	float reprojectionError;
	int bestSolution = 0;
	for (int solution = 0; solution < solutions; solution++) {
		combineVectors(translatedPoint, points + 3 * 3, positions + (long long) solution * 3, 1.0f, -1.0f, 3);
		computeReprojectionError(&reprojectionError, orientations + (long long) solution * 3 * 3, positions + (long long) solution * 3, pixels + 3 * 3, points + 3 * 3);
		if (reprojectionError < minimumError) {
			minimumError = reprojectionError;
			bestSolution = solution;
		}
	}
	memcpy(bestOrientation, orientations + (long long) bestSolution * 3 * 3, sizeof(float) * 3 * 3);
	memcpy(bestPosition, positions + (long long) bestSolution * 3, sizeof(float) * 3);
}

void computeReprojectionError(float *reprojectionError, float *orientation, float *position, float *pixel, float *point) {
	float reprojectedPixel[2];
	projectPoint(reprojectedPixel, orientation, position, point);
	*reprojectionError = (reprojectedPixel[0] - pixel[0]) * (reprojectedPixel[0] - pixel[0]) + (reprojectedPixel[1] - pixel[1]) * (reprojectedPixel[1] - pixel[1]);
	std::cout << std::setw(15) << reprojectedPixel[0] << std::setw(15) << reprojectedPixel[1] << std::setw(15) << *reprojectionError << "\n";
}

void projectPoint(float *pixel, float *orientation, float *position, float *point) {
	float translatedVector[3];
	combineVectors(translatedVector, point, position, 1.0f, -1.0f, 3);
	float z = matrixInnerProduct(translatedVector, orientation + 2, 3);
	pixel[0] = matrixInnerProduct(translatedVector, orientation, 3) / z;//matrixInnerProduct(translatedVector, orientation, 3) * HORIZONTAL_FOCAL_LENGTH / z + HORIZONTAL_PRINCIPAL_POINT
	pixel[1] = matrixInnerProduct(translatedVector, orientation + 1, 3) / z;//matrixInnerProduct(translatedVector, orientation + 1, 3) * VERTICAL_FOCAL_LENGTH / z + VERTICAL_PRINCIPAL_POINT
}

void unprojectPixel(float *ray, float *point) {
	ray[0] = point[0];//(point[0] - HORIZONTAL_PRINCIPAL_POINT) / HORIZONTAL_FOCAL_LENGTH
	ray[1] = point[1];//(point[1] - VERTICAL_PRINCIPAL_POINT) / VERTICAL_FOCAL_LENGTH
	ray[2] = 1.0f;
	normalizeVector(ray, 3);
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