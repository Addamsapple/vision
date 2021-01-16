//suggested camera frame: x axis - towards left of image, y axis - towards top of image, z axis, towards camera direction.

/*
TODO:
-solution pruning,
-angle determination
-substitute repeated products with power functions?
*/

//dont use chars where not necessary

#include <cmath>
#include <iostream>

const double fx = 1.0f;
const double fy = 1.0f;

const unsigned char REFINEMENT_ITERATIONS = 15;
const unsigned char FACTORIZATION_ITERATIONS = 10;

#define POLYNOMIAL_DEGREE 4

const float EPSILON = 0.01f;

//reorder method definitions

void computePolynomialRoots(float *roots, float *polynomial);
void refinePolynomialRoots(float *roots, float *polynomial);
void trilaterateCameraPosition(float *T, float *X, float *D, int configurations);
void computeQRFactorization(float *Q, float *R, float *A);

void computeCameraRotation(float *rotationMatrices, float *cameraPositions, float *featureWorldPositions, float *featureCameraPositions, int solutions);

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

void pose(float *I, float *X) {
	for (int point = 0; point < 3; point++) {
		I[point * 3] /= fx; I[point * 3 + 1] /= fy;
		I[point * 3 + 1] /= fy;
		normalizeVector(I + point * 3, 3);
	}
	float Cab = vectorDotProduct(I, I + 3, 3), Cac = vectorDotProduct(I, I + 6, 3), Cbc = vectorDotProduct(I + 3, I + 6, 3);
	float Rab = vectorDistance(X, X + 3, 3), Rac = vectorDistance(X, X + 6, 3), Rbc = vectorDistance(X + 3, X + 6, 3);
	float K1 = Rbc * Rbc / (Rac * Rac), K2 = Rbc * Rbc / (Rab * Rab);
	float *polynomial = new float[5];
	polynomial[4] = (K1 * K2 - K1 - K2) * (K1 * K2 - K1 - K2) - 4.0f * K1 * K2 * Cbc * Cbc;
	polynomial[0] = ((K1 * K2 + K1 - K2) * (K1 * K2 + K1 - K2) - 4.0f * K1 * K1 * K2 * Cac * Cac) / polynomial[4];
	polynomial[1] = (4.0f * (K1 * K2 + K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2.0f * K1 * K2 * Cab * Cac * Cac)) / polynomial[4];
	polynomial[2] = ((2.0f * K2 * (1 - K1) * Cab) * (2.0f * K2 * (1.0f - K1) * Cab) + 2.0f * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4.0f * K1 * ((K1 - K2) * Cbc * Cbc + K1 * (1.0f - K2) * Cac * Cac - 2.0f * (1.0f + K1) * K2 * Cab * Cac * Cbc)) / polynomial[4];
	polynomial[3] = (4.0f * (K1 * K2 - K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2.0f * K2 * Cab * Cbc)) / polynomial[4];
	polynomial[4] = 1.0f;
	float *roots = new float[4];
	computePolynomialRoots(roots, polynomial);
	refinePolynomialRoots(roots, polynomial);
	float *D = new float[4 * 3];
	int configurations = 0;
	for (int root = 0; root < 4; root++) {
		float x = roots[root];
		float xx = x * x;
		float xxx = xx * x;
		float xxxx = xxx * x;
		if (std::abs(polynomial[4] * xxxx + polynomial[3] * xxx + polynomial[2] * xx + polynomial[1] * x + polynomial[0]) < EPSILON) {
			if (root > 0 && std::abs(roots[root - 1] - x) < EPSILON)
				continue;
			float a = Rab / std::sqrt(xx - 2.0f * x * Cab + 1.0f);
			float b = a * x;
			float m = 1.0f - K1, mp = 1.0f;
			float q = xx - K1, qp = (1.0f - K2) * xx + 2.0f * K2 * Cab * x - K2;
			if (std::abs(m * qp - mp * q) > EPSILON) {
				float p = 2.0f * (K1 * Cac - Cbc * x), pp = -2.0f * Cbc * x;
				float y = (pp * q - p * qp) / (m * qp - mp * q);
				float c = a * y;
				D[configurations * 3] = a; D[configurations * 3 + 1] = b; D[configurations * 3 + 2] = c;
				configurations++;
			} else {
				float y[2] = {Cac - std::sqrt(Cac * Cac + Rac * Rac / (a * a) - 1.0f), 0.0f}; y[1] = 2.0f * Cac - y[0];
				float c[2] = {a * y[0], a * y[1]};
				for (int index = 0; index < 2; index++)
					if (std::abs(Rbc * Rbc - b * b - c[index] * c[index] + 2.0f * b * c[index] * Cbc) < EPSILON) {
						D[configurations * 3] = a; D[1 + configurations * 3] = b; D[2 + configurations * 3] = c[index];//change indexing order
						configurations++;
					}
			}
		}
	}
	float *T = new float[(long long) configurations * 2 * 3];
	trilaterateCameraPosition(T, X, D, configurations);
	//
	std::cout << T[0] << " " << T[1] << " " << T[2] << "\n";
	std::cout << T[3] << " " << T[4] << " " << T[5] << "\n";
	std::cout << T[6] << " " << T[7] << " " << T[8] << "\n";
	std::cout << T[9] << " " << T[10] << " " << T[11] << "\n";
	//
	float *XT = new float[3 * 3];
	float *V = new float[3 * 3];
	float *R = new float[3 * 3];
	/*for (int solution = 0; solution < configurations * 2; solution++) {
		combineVectors(XT, X, T + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(XT + 3, X + 3, T + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(XT + 2 * 3, X + 2 * 3, T + (long long) solution * 3, 1.0f, -1.0f, 3);
		scaleVector(V, I, vectorMagnitude(XT, 3), 3);
		scaleVector(V + 3, I + 3, vectorMagnitude(XT + 3, 3), 3);
		scaleVector(V + 2 * 3, I + 2 * 3, vectorMagnitude(XT + 2 * 3, 3), 3);
		invertMatrix(XT, 3);
		multiplyMatrices(R, XT, V, 3);
		std::cout << "\n";
		std::cout << R[0] << " " << R[1] << " " << R[2] << "\n";
		std::cout << R[3] << " " << R[4] << " " << R[5] << "\n";
		std::cout << R[6] << " " << R[7] << " " << R[8] << "\n";
	}*/

	computeCameraRotation(R, T, X, I, 4);

	delete[] polynomial; delete[] roots; delete[] D; delete[] T; delete[] XT; delete[] V; delete[] R;
}

void computeCameraRotation(float *rotationMatrices, float *cameraPositions, float *featureWorldPositions, float *featureCameraPositions, int solutions) {
	float translatedFeatureWorldPositions[3 * 3];
	float scaledFeatureCameraPositions[3 * 3];
	for (int solution = 0; solution < solutions; solution++) {
		combineVectors(translatedFeatureWorldPositions, featureWorldPositions, cameraPositions + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedFeatureWorldPositions + 3, featureWorldPositions + 3, cameraPositions + (long long) solution * 3, 1.0f, -1.0f, 3);
		combineVectors(translatedFeatureWorldPositions + 2 * 3, featureWorldPositions + 2 * 3, cameraPositions + (long long) solution * 3, 1.0f, -1.0f, 3);
		scaleVector(scaledFeatureCameraPositions, featureCameraPositions, vectorMagnitude(translatedFeatureWorldPositions, 3), 3);
		scaleVector(scaledFeatureCameraPositions + 3, featureCameraPositions + 3, vectorMagnitude(translatedFeatureWorldPositions + 3, 3), 3);
		scaleVector(scaledFeatureCameraPositions + 2 * 3, featureCameraPositions + 2 * 3, vectorMagnitude(translatedFeatureWorldPositions + 2 * 3, 3), 3);
		invertMatrix(translatedFeatureWorldPositions, 3);
		multiplyMatrices(rotationMatrices, translatedFeatureWorldPositions, scaledFeatureCameraPositions, 3);
		transposeMatrix(rotationMatrices, 3);
		std::cout << "\n";
		std::cout << rotationMatrices[0] << " " << rotationMatrices[1] << " " << rotationMatrices[2] << "\n";
		std::cout << rotationMatrices[3] << " " << rotationMatrices[4] << " " << rotationMatrices[5] << "\n";
		std::cout << rotationMatrices[6] << " " << rotationMatrices[7] << " " << rotationMatrices[8] << "\n";
	
	}
}

void computePolynomialRoots(float *roots, float *polynomialCoefficients) {
	float matrixA[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixQ[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	float matrixR[POLYNOMIAL_DEGREE * POLYNOMIAL_DEGREE];
	for (int row = 0; row < POLYNOMIAL_DEGREE - 1; row++) {
		for (int column = 0; column < row + 1; column++)
			matrixA[column + row * POLYNOMIAL_DEGREE] = 0;
		matrixA[row + 1 + row * POLYNOMIAL_DEGREE] = 1;
		for (int column = row + 2; column < POLYNOMIAL_DEGREE; column++)
			matrixA[column + row * POLYNOMIAL_DEGREE] = 0;
	}
	for (int column = 0; column < POLYNOMIAL_DEGREE; column++)
		matrixA[column + (POLYNOMIAL_DEGREE - 1) * POLYNOMIAL_DEGREE] = -polynomialCoefficients[column];
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

void refinePolynomialRoots(float *roots, float *polynomialCoefficients) {
	float rootPowers[POLYNOMIAL_DEGREE + 1];
	rootPowers[0] = 1.0f;
	for (int root = 0; root < POLYNOMIAL_DEGREE; root++)
		for (int iteration = 0; iteration < REFINEMENT_ITERATIONS; iteration++) {
			for (int power = 1; power < POLYNOMIAL_DEGREE + 1; power++)
				rootPowers[power] = rootPowers[power - 1] * roots[root];
			float function = 0.0f;
			float firstFunctionDerivative = 0.0f;
			float secondFunctionDerivative = 0.0f;
			for (int term = 0; term < POLYNOMIAL_DEGREE + 1; term++)
				function += polynomialCoefficients[term] * rootPowers[term];
			for (int term = 1; term < POLYNOMIAL_DEGREE + 1; term++)
				firstFunctionDerivative += term * polynomialCoefficients[term] * rootPowers[term - 1];
			for (int term = 2; term < POLYNOMIAL_DEGREE + 1; term++)
				secondFunctionDerivative += term * (term - 1) * polynomialCoefficients[term] * rootPowers[term - 2];
			roots[root] -= function * firstFunctionDerivative / (firstFunctionDerivative * firstFunctionDerivative - function * secondFunctionDerivative * 0.5f);
		}
}

void trilaterateCameraPosition(float *cameraPositions, float *featurePositions, float *distances, int configurations) {
	float vectorB[3];
	combineVectors(vectorB, featurePositions + 3, featurePositions, 1.0f, -1.0f, 3);
	float vectorC[3];
	combineVectors(vectorC, featurePositions + 2 * 3, featurePositions, 1.0f, -1.0f, 3);
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
		float camereXCoordinate = (distances[configuration * 3] * distances[configuration * 3] - distances[1 + configuration * 3] * distances[1 + configuration * 3] + vectorBXComponent * vectorBXComponent) / (2.0f * vectorBXComponent);
		float cameraYCoordinate = (distances[configuration * 3] * distances[configuration * 3] - distances[2 + configuration * 3] * distances[2 + configuration * 3] + vectorCXComponent * vectorCXComponent + vectorCYComponent * vectorCYComponent - 2.0f * vectorCXComponent * camereXCoordinate) / (2.0f * vectorCYComponent);
		float cameraZCoordinate = std::sqrt(distances[configuration * 3] * distances[configuration * 3] - camereXCoordinate * camereXCoordinate - cameraYCoordinate * cameraYCoordinate);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3, featurePositions, frameMatrix, 1.0f, camereXCoordinate, 3);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3, cameraPositions + (long long) configuration * 2 * 3, frameMatrix + 3, 1.0f, cameraYCoordinate, 3);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3, cameraPositions + (long long) configuration * 2 * 3, frameMatrix + 2 * 3, 1.0f, cameraZCoordinate, 3);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3 + 3, featurePositions, frameMatrix, 1.0f, camereXCoordinate, 3);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3 + 3, cameraPositions + (long long) configuration * 2 * 3 + 3, frameMatrix + 3, 1.0f, cameraYCoordinate, 3);
		combineVectors(cameraPositions + (long long) configuration * 2 * 3 + 3, cameraPositions + (long long) configuration * 2 * 3 + 3, frameMatrix + 2 * 3, 1.0f, -cameraZCoordinate, 3);
	}
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