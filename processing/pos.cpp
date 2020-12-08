/*
Written by Justin Tulleken.

*/

//suggested camera frame: x axis - towards left of image, y axis - towards top of image, z axis, towards camera direction.

/*
TODO:
-solution pruning,
-angle determination
-substitute repeated products with power functions?
*/

#include <cmath>
#include <iostream>

const double fx = 1.0f;
const double fy = 1.0f;

const unsigned char ITERATIONS_HALLEY = 15;
const unsigned char ITERATIONS_QR = 10;

const float EPSILON = 0.01f;

void solve(float *roots, float *polynomial, unsigned char degree);
void factorize(float *A, float *Q, float *R, unsigned char size);
void polish(float *roots, float *polynomial, unsigned char degree);
void trilaterate(float *T, float *X, float *D, unsigned char configurations);
void transpose(float *A, unsigned char size);
void invert(float *A, unsigned char size);
void multiply(float *A, float *B, float *C, unsigned char size);
void normalize(float *A, unsigned char size);
void scale(float *A, float *B, float b, unsigned char size);
void combine(float *A, float *B, float *C, float b, float c, unsigned char size);
void crossMultiply(float *A, float *B, float *C);
float innerProduct(float *A, float *B, unsigned char size);
float dotProduct(float *A, float *B, unsigned char size);
float distance(float *A, float *B, unsigned char size);
float magnitude(float *A, unsigned char size);

void pose(float *I, float *X) {
	for (unsigned char point = 0; point < 3; point++) {
		I[point * 3] /= fx; I[point * 3 + 1] /= fy;
		I[point * 3 + 1] /= fy;
		normalize(I + point * 3, 3);
	}
	float Cab = dotProduct(I, I + 3, 3), Cac = dotProduct(I, I + 6, 3), Cbc = dotProduct(I + 3, I + 6, 3);
	float Rab = distance(X, X + 3, 3), Rac = distance(X, X + 6, 3), Rbc = distance(X + 3, X + 6, 3);
	float K1 = Rbc * Rbc / (Rac * Rac), K2 = Rbc * Rbc / (Rab * Rab);
	float *polynomial = new float[5];
	polynomial[4] = (K1 * K2 - K1 - K2) * (K1 * K2 - K1 - K2) - 4.0f * K1 * K2 * Cbc * Cbc;
	polynomial[0] = ((K1 * K2 + K1 - K2) * (K1 * K2 + K1 - K2) - 4.0f * K1 * K1 * K2 * Cac * Cac) / polynomial[4];
	polynomial[1] = (4.0f * (K1 * K2 + K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2.0f * K1 * K2 * Cab * Cac * Cac)) / polynomial[4];
	polynomial[2] = ((2.0f * K2 * (1 - K1) * Cab) * (2.0f * K2 * (1.0f - K1) * Cab) + 2.0f * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4.0f * K1 * ((K1 - K2) * Cbc * Cbc + K1 * (1.0f - K2) * Cac * Cac - 2.0f * (1.0f + K1) * K2 * Cab * Cac * Cbc)) / polynomial[4];
	polynomial[3] = (4.0f * (K1 * K2 - K1 - K2) * K2 * (1.0f - K1) * Cab + 4.0f * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2.0f * K2 * Cab * Cbc)) / polynomial[4];
	polynomial[4] = 1.0f;
	float *roots = new float[4];
	solve(roots, polynomial, 4);
	polish(roots, polynomial, 4);
	float *D = new float[4 * 3];
	unsigned char configurations = 0;
	for (unsigned char root = 0; root < 4; root++) {
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
				for (unsigned char index = 0; index < 2; index++)
					if (std::abs(Rbc * Rbc - b * b - c[index] * c[index] + 2.0f * b * c[index] * Cbc) < EPSILON) {
						D[configurations * 3] = a; D[configurations * 3 + 1] = b; D[configurations * 3 + 2] = c[index];
						configurations++;
					}
			}
		}
	}
	float *T = new float[configurations * 2 * 3];
	trilaterate(T, X, D, configurations);
	float *XT = new float[3 * 3];
	float *V = new float[3 * 3];
	float *R = new float[3 * 3];
	for (unsigned char solution = 0; solution < configurations * 2; solution++) {
		combine(XT, X, T + solution * 3, 1.0f, -1.0f, 3);
		combine(XT + 3, X + 3, T + solution * 3, 1.0f, -1.0f, 3);
		combine(XT + 6, X + 6, T + solution * 3, 1.0f, -1.0f, 3);
		scale(V, I, magnitude(XT, 3), 3);
		scale(V + 3, I + 3, magnitude(XT + 3, 3), 3);
		scale(V + 6, I + 6, magnitude(XT + 6, 3), 3);
		invert(XT, 3);
		multiply(R, XT, V, 3);
	}
	delete[] polynomial; delete[] roots; delete[] D; delete[] T; delete[] XT; delete[] V; delete[] R;
}

void solve(float *roots, float *polynomial, unsigned char degree) {
	float *A = new float[degree * degree];
	float *Q = new float[degree * degree];
	float *R = new float[degree * degree];
	for (unsigned char row = 0; row < degree - 1; row++) {
		for (unsigned char column = 0; column < row + 1; column++)
			A[row * degree + column] = 0;
		A[row * degree + row + 1] = 1;
		for (unsigned char column = row + 2; column < degree; column++)
			A[row * degree + column] = 0;
	}
	for (unsigned char column = 0; column < degree; column++)
		A[(degree - 1) * degree + column] = -polynomial[column];
	for (unsigned char iteration = 0; iteration < ITERATIONS_QR; iteration++) {
		factorize(A, Q, R, degree);
		multiply(A, R, Q, degree);
	}
	for (unsigned root = 0; root < degree; root++)
		roots[root] = A[root * degree + root];
	delete[] A;	delete[] Q;	delete[] R;
}

void factorize(float *A, float *Q, float *R, unsigned char size) {
	for (unsigned char axisQ = 0; axisQ < size; axisQ++) {
		for (unsigned char element = 0; element < size; element++)
			Q[axisQ * size + element] = A[element * size + axisQ];
		for (unsigned char axisA = 0; axisA < axisQ; axisA++)
			combine(Q + axisQ * size, Q + axisQ * size, Q + axisA * size, 1.0f, -innerProduct(Q + axisA * size, A + axisQ, size), size);
		normalize(Q + axisQ * size, size);
	}
	multiply(R, Q, A, size);
	transpose(Q, size);
}

//rename to refine?
void polish(float *roots, float *polynomial, unsigned char degree) {
	float *x = new float[degree + 1];
	x[0] = 1.0f;
	for (unsigned char root = 0; root < degree; root++)
		for (int iteration = 0; iteration < ITERATIONS_HALLEY; iteration++) {
			for (unsigned char power = 1; power < degree + 1; power++)
				x[power] = x[power - 1] * roots[root];
			float gx = 0, gpx = 0, gppx = 0;
			for (unsigned char term = 0; term < degree + 1; term++)
				gx += polynomial[term] * x[term];
			for (unsigned char term = 1; term < degree + 1; term++)
				gpx += term * polynomial[term] * x[term - 1];
			for (unsigned char term = 2; term < degree + 1; term++)
				gppx += term * (term - 1) * polynomial[term] * x[term - 2];
			roots[root] -= gx * gpx / (gpx * gpx - 0.5f * gx * gppx);
		}
	delete[] x;
}

void trilaterate(float *T, float *X, float *D, unsigned char configurations) {
	float *ba = new float[3];
	combine(ba, X + 3, X, 1.0f, -1.0f, 3);
	float *ca = new float[3];
	combine(ca, X + 6, X, 1.0f, -1.0f, 3);
	float *frame = new float[3 * 3];
	scale(frame, ba, 1.0f, 3);
	normalize(frame, 3);
	combine(frame + 3, ca, frame, 1.0f, -dotProduct(ca, frame, 3), 3);
	normalize(frame + 3, 3);
	crossMultiply(frame + 6, frame, frame + 3);
	float bx = magnitude(ba, 3);
	float cx = dotProduct(ca, frame, 3);
	float cy = dotProduct(ca, frame + 3, 3);
	for (unsigned char configuration = 0; configuration < configurations; configuration++) {
		float x = (D[configuration * 3] * D[configuration * 3] - D[configuration * 3 + 1] * D[configuration * 3 + 1] + bx * bx) / (2 * bx);
		float y = (D[configuration * 3] * D[configuration * 3] - D[configuration * 3 + 2] * D[configuration * 3 + 2] + cx * cx + cy * cy - 2 * cx * x) / (2 * cy);
		float z = std::sqrt(D[configuration * 3] * D[configuration * 3] - x * x - y * y);
		combine(T + configuration * 2 * 3, X, frame, 1.0f, x, 3);
		combine(T + configuration * 2 * 3, T + configuration * 2 * 3, frame + 3, 1.0f, y, 3);
		combine(T + configuration * 2 * 3, T + configuration * 2 * 3, frame + 6, 1.0f, z, 3);
		combine(T + configuration * 2 * 3 + 3, X, frame, 1.0f, x, 3);
		combine(T + configuration * 2 * 3 + 3, T + configuration * 2 * 3 + 3, frame + 3, 1.0f, y, 3);
		combine(T + configuration * 2 * 3 + 3, T + configuration * 2 * 3 + 3, frame + 6, 1.0f, -z, 3);
	}
	delete[] ba; delete[] ca; delete[] frame;
}

void transpose(float *A, unsigned char size) {
	for (unsigned char row = 0; row < size; row++)
		for (unsigned char column = row + 1; column < size; column++) {
			float temporary = A[row * size + column];
			A[row * size + column] = A[column * size + row];
			A[column * size + row] = temporary;
		}
}

//An Efficient and Simple Algorithm for Matrix Inversion, by Ahmad Farooq and Khan Hamid
void invert(float *A, unsigned char size) {
	for (unsigned char iteration = 0; iteration < size; iteration++) {
		float pivot = A[iteration * size + iteration];
		for (unsigned char row = 0; row < size; row++)
			A[row * size + iteration] = -A[row * size + iteration] / pivot;
		for (unsigned char row = 0; row < size; row++)
			if (row != iteration)
				for (unsigned char column = 0; column < size; column++)
					if (column != iteration)
						A[row * size + column] = A[row * size + column] + A[iteration * size + column] * A[row * size + iteration];
		for (unsigned char column = 0; column < size; column++)
			A[iteration * size + column] = A[iteration * size + column] / pivot;
		A[iteration * size + iteration] = 1.0f / pivot;
	}
}

void multiply(float *A, float *B, float *C, unsigned char size) {
	for (unsigned char row = 0; row < size; row++)
		for (unsigned char column = 0; column < size; column++)
			A[row * size + column] = innerProduct(B + row * size, C + column, size);
}

void normalize(float *A, unsigned char size) {
	float length = magnitude(A, size);
	for (unsigned char element = 0; element < size; element++)
		A[element] /= length;
}

void scale(float *A, float *B, float b, unsigned char size) {
	for (unsigned char element = 0; element < size; element++)
		A[element] = b * B[element];
}

void combine(float *A, float *B, float *C, float b, float c, unsigned char size) {
	for (unsigned char element = 0; element < size; element++)
		A[element] = b * B[element] + c * C[element];
}

void crossMultiply(float *A, float *B, float *C) {
	A[0] = B[1] * C[2] - B[2] * C[1];
	A[1] = B[2] * C[0] - B[0] * C[2];
	A[2] = B[0] * C[1] - B[1] * C[0];
}
	
float innerProduct(float *A, float *B, unsigned char size) {
	float result = 0.0f;
	for (unsigned char element = 0; element < size; element++)
		result += A[element] * B[element * size];
	return result;
}

float dotProduct(float *A, float *B, unsigned char size) {
	float result = 0.0f;
	for (unsigned char element = 0; element < size; element++)
		result += A[element] * B[element];
	return result;
}

float distance(float *A, float *B, unsigned char size) {
	float result = 0.0f;
	for (unsigned char element = 0; element < size; element++) {
		float difference = A[element] - B[element];
		result += difference * difference;
	}
	return std::sqrt(result);
}

float magnitude(float *A, unsigned char size) {
	float result = 0.0f;
	for (unsigned char element = 0; element < size; element++)
		result += A[element] * A[element];
	return std::sqrt(result);
}