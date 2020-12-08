#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include "png.hpp"

//although texture usage doesnt improve speed, can make code simpler, since it has built in boundary handling (i.e. texture wrapping, etc.)

/*
NOTES ON PERFORMANCE
-memory copy speed is significantly improved through use of cudaMallocPitch/cudaMemcpy2D
-memory copy speed might be further improved by using pinned memory.
-use of a texture for the image halves the computation time for sumVertically, but it is not worth it,
	given that sumHorizontally is the bottleneck, and that the texture decreases integration accuracy.
-accesses to constant memory can be avoided by expressing determining octave/scale through equations.
-one would expect supress to be slower than compute, given that it requires more potential memory accesses per thread,
	however, it seems that early returns greatly reduce the average number of memory accesses.
-determinant accesses during suppression can be reduced by using shared memory, assuming threads are allocated to rows, not pixels.
*/

//change descriptor to return sign of laplacian as well, which will improve matching times

#define SCALES 10
#define PADDING 245
#define W 0.912f

#define MAX_KEYPOINTS 1000

#define DESCRIPTOR_WIDTH 10

#define SIGMA 0.4f

#define EPSILON 0.00001f

#define DETECTION_THRESHOLD 10000.0f

#define MATCHING_DISTANCE_THRESHOLD 0.1f

#define MATCHING_RATIO_THRESHOLD 0.5f

__constant__ int d_co[SCALES] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3};
__constant__ int d_so[SCALES] = {0, 0, 0, 1, 1, 2, 2, 3, 3, 3};
__constant__ int d_no[SCALES * 3] = {0, 0, 0, -1, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, 0, 0, 0};
__constant__ int d_l[SCALES] = {3, 5, 7, 9, 13, 17, 25, 33, 49, 65};

__constant__ float d_g[12][12] = {
	0.014614763f, 0.013958917f, 0.012162744f, 0.00966788f, 0.00701053f, 0.004637568f, 0.002798657f, 0.001540738f, 0.000773799f, 0.000354525f, 0.000148179f, 0.0f,
	0.013958917f, 0.013332502f, 0.011616933f, 0.009234028f, 0.006695928f, 0.004429455f, 0.002673066f, 0.001471597f, 0.000739074f, 0.000338616f, 0.000141529f, 0.0f,
	0.012162744f, 0.011616933f, 0.010122116f, 0.008045833f, 0.005834325f, 0.003859491f, 0.002329107f, 0.001282238f, 0.000643973f, 0.000295044f, 0.000123318f, 0.0f,
	0.00966788f, 0.009234028f, 0.008045833f, 0.006395444f, 0.004637568f, 0.003067819f, 0.001851353f, 0.001019221f, 0.000511879f, 0.000234524f, 9.80224E-05f, 0.0f,
	0.00701053f, 0.006695928f, 0.005834325f, 0.004637568f, 0.003362869f, 0.002224587f, 0.001342483f, 0.000739074f, 0.000371182f, 0.000170062f, 7.10796E-05f, 0.0f,
	0.004637568f, 0.004429455f, 0.003859491f, 0.003067819f, 0.002224587f, 0.001471597f, 0.000888072f, 0.000488908f, 0.000245542f, 0.000112498f, 4.70202E-05f, 0.0f,
	0.002798657f, 0.002673066f, 0.002329107f, 0.001851353f, 0.001342483f, 0.000888072f, 0.000535929f, 0.000295044f, 0.000148179f, 6.78899E-05f, 2.83755E-05f, 0.0f,
	0.001540738f, 0.001471597f, 0.001282238f, 0.001019221f, 0.000739074f, 0.000488908f, 0.000295044f, 0.00016243f, 8.15765E-05f, 3.73753E-05f, 1.56215E-05f, 0.0f,
	0.000773799f, 0.000739074f, 0.000643973f, 0.000511879f, 0.000371182f, 0.000245542f, 0.000148179f, 8.15765E-05f, 4.09698E-05f, 1.87708E-05f, 7.84553E-06f, 0.0f,
	0.000354525f, 0.000338616f, 0.000295044f, 0.000234524f, 0.000170062f, 0.000112498f, 6.78899E-05f, 3.73753E-05f, 1.87708E-05f, 8.60008E-06f, 3.59452E-06f, 0.0f,
	0.000148179f, 0.000141529f, 0.000123318f, 9.80224E-05f, 7.10796E-05f, 4.70202E-05f, 2.83755E-05f, 1.56215E-05f, 7.84553E-06f, 3.59452E-06f, 1.50238E-06f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

__device__ void reduce(volatile float *s_d);
__device__ float filter(int *d_U, int d_w, int d_r, int d_c, int d_Lx, int d_Ly);
__device__ float dx(int *d_U, int d_w, int d_r, int d_c, int d_L);
__device__ float dy(int *d_U, int d_w, int d_r, int d_c, int d_L);
__device__ float dxx(int *d_U, int d_w, int d_r, int d_c, int d_L);
__device__ float dyy(int *d_U, int d_w, int d_r, int d_c, int d_L);
__device__ float dxy(int *d_U, int d_w, int d_r, int d_c, int d_L);

__global__ void compute(float *d_H, int *d_U, int d_w, int d_h) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / d_w;
	l_c -= l_r * d_w;
	if (l_c > PADDING - 1 && l_c < d_w - PADDING && l_r > PADDING - 1 && l_r < d_h - PADDING) {
		//#pragma unroll
		for (int l_s = 0; l_s < SCALES; l_s++) {
			int l_i = (1 << d_co[l_s]);
			if ((l_c - PADDING) % l_i != 0 || (l_r - PADDING) % l_i != 0)
				return;
			float l_xxyy = 1.0f / d_l[l_s];
			l_xxyy *= l_xxyy;
			float l_xyxy = W * l_xxyy;
			l_xxyy *= l_xxyy;
			l_xxyy *= dxx(d_U, d_w, l_r, l_c, d_l[l_s]) * dyy(d_U, d_w, l_r, l_c, d_l[l_s]);
			l_xyxy *= dxy(d_U, d_w, l_r, l_c, d_l[l_s]);
			l_xyxy *= l_xyxy;
			d_H[l_c + l_r * d_w + l_s * d_w * d_h] = l_xxyy - l_xyxy;
		}
	}
}

__global__ void supress(int *d_k, float *d_H, int d_w, int d_h, unsigned char *d_I, int d_IP) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / d_w;
	l_c -= l_r * d_w;
	if (l_c > PADDING - 1 && l_c < d_w - PADDING && l_r > PADDING - 1 && l_r < d_h - PADDING) {
		for (int l_s = 1; l_s < SCALES - 1; l_s++) {
			int l_i = 1 << d_so[l_s];
			if ((l_c - PADDING) % l_i != 0 || (l_r - PADDING) % l_i != 0)
				return;
			float l_d = d_H[l_c + l_r * d_w + l_s * d_w * d_h];
			if (l_d < DETECTION_THRESHOLD)
				continue;
			float l_m = FLT_MIN;
			for (int l_ds = 0; l_ds < 3; l_ds++)
				for (int l_dr = -l_i; l_dr < l_i + 1 && l_d >= l_m; l_dr += l_i)
					#pragma unroll
					for (int l_dc = -l_i; l_dc < l_i + 1 && l_d >= l_m; l_dc += l_i)
						l_m = fmaxf(l_m, d_H[l_c + l_dc + (l_r + l_dr) * d_w + (l_s + d_no[l_s * 3 + l_ds]) * d_w * d_h]);
			if (l_d == l_m) {
				int l_n = atomicAdd(d_k, 1);
				if (l_n < MAX_KEYPOINTS) {
					d_H[l_n + (d_h - PADDING) * d_w] = l_r;
					d_H[l_n + MAX_KEYPOINTS + (d_h - PADDING) * d_w] = l_c;
					d_H[l_n + MAX_KEYPOINTS * 2 + (d_h - PADDING) * d_w] = l_s;
				}
			}
		}
	}
}

#define H(dr, dc, ds) d_H[l_c + dc + (l_r + dr) * d_w + (l_s + d_no[l_s * 3 + ds + 1]) * d_w * d_h]

__global__ void refine(float *d_H, int *d_r, int d_w, int d_h, int d_k) {
	int l_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_id > d_k - 1)
		return;
	int l_r = d_H[l_id + (d_h - PADDING) * d_w];
	int l_c = d_H[l_id + MAX_KEYPOINTS + (d_h - PADDING) * d_w];
	int l_s = d_H[l_id + MAX_KEYPOINTS * 2 + (d_h - PADDING) * d_w];
	int l_i = 1 << d_so[l_s];
	float l_m[12];
	l_m[3] = (H(0, l_i, 0) - H(0, -l_i, 0)) / (2.0f * l_i);
	l_m[7] = (H(l_i, 0, 0) - H(-l_i, 0, 0)) / (2.0f * l_i);
	l_m[11] = (H(0, 0, 1) - H(0, 0, -1)) / (4.0f * l_i);
	l_m[0] = (H(0, l_i, 0) - 2.0f * H(0, 0, 0) + H(0, -l_i, 0)) / (l_i * l_i);
	l_m[5] = (H(l_i, 0, 0) - 2.0f * H(0, 0, 0) + H(-l_i, 0, 0)) / (l_i * l_i);
	l_m[10] = (H(0, 0, 1) - 2.0f * H(0, 0, 0) + H(0, 0, -1)) / (4.0f * l_i * l_i);
	l_m[1] = (H(l_i, l_i, 0) + H(-l_i, -l_i, 0) - H(l_i, -l_i, 0) - H(-l_i, l_i, 0)) / (4.0f * l_i * l_i);
	l_m[2] = (H(0, l_i, 1) + H(0, -l_i, -1) - H(0, -l_i, 1) - H(0, l_i, -1)) / (8.0f * l_i * l_i);
	l_m[6] = (H(l_i, 0, 1) + H(-l_i, 0, -1) - H(-l_i, 0, 1) - H(l_i, 0, -1)) / (8.0f * l_i * l_i);
	l_m[4] = l_m[1];
	l_m[8] = l_m[2];
	l_m[9] = l_m[6];
	int l_y, l_z;
	for (int l_ro = 0; l_ro < 3; l_ro++) {
		float l_d = l_m[l_ro * 4];
		if (fabs(l_d) > EPSILON)
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + l_ro * 4] /= l_d;
	}
	if (fabsf(l_m[0]) > EPSILON) {
		if (fabsf(l_m[4]) > EPSILON)
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + 4] -= l_m[l_co];
		if (fabsf(l_m[8]) > EPSILON)
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + 8] -= l_m[l_co];
		l_y = 1;
		l_z = 2;
	} else if (fabsf(l_m[4]) > EPSILON) {
		if (fabsf(l_m[8]) > EPSILON)
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + 8] -= l_m[l_co + 4];
		l_y = 0;
		l_z = 2;
	} else if (fabsf(l_m[8]) > EPSILON) {
		l_y = 0;
		l_z = 1;
	} else
		return;
	if (fabsf(l_m[1 + l_z * 4]) > EPSILON) {
		if (fabs(l_m[1 + l_y * 4]) > EPSILON) {
			float l_d = l_m[1 + l_y * 4];
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + l_y * 4] /= l_d;
			l_d = l_m[1 + l_z * 4];
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + l_z * 4] /= l_d;
			for (int l_co = 0; l_co < 4; l_co++)
				l_m[l_co + l_z * 4] -= l_m[l_co + l_y * 4];
		} else {
			int l_t = l_y;
			l_y = l_z;
			l_z = l_t;
		}
	} else if (!(fabs(l_m[1 + l_y * 4]) > EPSILON))
		return;
	float l_ds = l_m[3 + l_z * 4] / l_m[2 + l_z * 4];
	float l_dc = (l_m[3 + l_y * 4] - l_m[2 + l_y * 4] * l_ds) / l_m[1 + l_y * 4];
	float l_dr = (l_m[3 + (3 - l_y - l_z) * 4] - l_m[1 + (3 - l_y - l_z) * 4] * l_dc - l_m[2 + (3 - l_y - l_z) * 4] * l_ds) / l_m[(3 - l_y - l_z) * 4];
	if (fmaxf(fmaxf(fabsf(l_dr), fabsf(l_dc)), 0.5f * fabsf(l_ds)) < l_i) {
		float l_nr = l_r - l_dr;
		float l_nc = l_c - l_dc;
		float l_ns = d_l[l_s] - l_ds;
		int l_d = llrintf(l_ns * (DESCRIPTOR_WIDTH * SIGMA + 1.0f)) + 1;
		if (l_c - l_d > -EPSILON && l_c + l_d < d_w + EPSILON && l_r - l_d > -EPSILON && l_r + l_d < d_h + EPSILON) {
			int l_n = atomicAdd(d_r, 1);
			d_H[l_n * 3]  = l_nr;
			d_H[1 + l_n * 3] = l_nc;
			d_H[2 + l_n * 3] = l_ns;
		}
	}
}

#define square(n) n * n

//rename d_H in this context
__global__ void describe(float *d_D, float *d_H, int *d_U, int d_w) {
	int l_br = threadIdx.z / 4;
	int l_bc = threadIdx.z - l_br * 4;
	int l_rr = threadIdx.y + l_br * 5 - DESCRIPTOR_WIDTH;
	int l_rc = threadIdx.x + l_bc * 5 - DESCRIPTOR_WIDTH;
	float l_s = d_H[2 + blockIdx.x * 3] * SIGMA;
	int l_sr = llrintf(d_H[blockIdx.x * 3] + l_s * l_rr);
	int l_sc = llrintf(d_H[1 + blockIdx.x * 3] + l_s * l_rc);
	float l_g = d_g[abs(l_rr)][abs(l_rc)];
	__shared__ float s_dx[16][5][5];
	__shared__ float s_dy[16][5][5];
	s_dx[threadIdx.z][threadIdx.x][threadIdx.y] = dx(d_U, d_w, l_sr, l_sc, llrintf(l_s)) * l_g;
	s_dy[threadIdx.z][threadIdx.x][threadIdx.y] = dy(d_U, d_w, l_sr, l_sc, llrintf(l_s)) * l_g;
	__syncthreads();
	__shared__ float s_s[16][4][5];
	if (threadIdx.y == 0) {
		float l_sx = 0.0f, l_sy = 0.0f, l_sax = 0.0f, l_say = 0.0f;
		for (int l_r = 0; l_r < 5; l_r++) {
			l_sx += s_dx[threadIdx.z][threadIdx.x][l_r];
			l_sy += s_dy[threadIdx.z][threadIdx.x][l_r];
			l_sax += fabsf(s_dx[threadIdx.z][threadIdx.x][l_r]);
			l_say += fabsf(s_dy[threadIdx.z][threadIdx.x][l_r]);	
		}
		s_s[threadIdx.z][0][threadIdx.x] = l_sx;
		s_s[threadIdx.z][1][threadIdx.x] = l_sy;
		s_s[threadIdx.z][2][threadIdx.x] = l_sax;
		s_s[threadIdx.z][3][threadIdx.x] = l_say;
	}
	__syncthreads();
	__shared__ float s_d[16][4];
	if (threadIdx.x < 4 && threadIdx.y == 0) {
		float l_c = s_s[threadIdx.z][threadIdx.x][0] + s_s[threadIdx.z][threadIdx.x][1] + s_s[threadIdx.z][threadIdx.x][2] + s_s[threadIdx.z][threadIdx.x][3];
		s_d[threadIdx.z][threadIdx.x] = l_c;
		d_D[threadIdx.x + threadIdx.z * 4 + blockIdx.x * 64] = l_c;
	}
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
		d_H[threadIdx.z + blockIdx.x * 16 + MAX_KEYPOINTS * 3] = square(s_d[threadIdx.z][0]) + square(s_d[threadIdx.z][1]) + square(s_d[threadIdx.z][2]) + square(s_d[threadIdx.z][3]);
}

//potential occupancy can be doubled by assigning 2 key points to each block.
__global__ void normalize(float *d_D, float *d_H) {
	int l_ei = threadIdx.x * 4 + blockIdx.x * 16;
	__shared__ float s_s[4];
	if (threadIdx.x < 4)
		s_s[threadIdx.x] = d_H[l_ei + MAX_KEYPOINTS * 3] + d_H[l_ei + 1 + MAX_KEYPOINTS * 3] + d_H[l_ei + 2 + MAX_KEYPOINTS * 3] + d_H[l_ei + 3 + MAX_KEYPOINTS * 3];
	__syncthreads();
	float s_d = rsqrtf(s_s[0] + s_s[1] + s_s[2] + s_s[3]);
	d_D[threadIdx.x + blockIdx.x * 64] *= s_d;
}

//need to consider this: features detected between views MUST ALSO BE DETECTED BETWEEN FRAMES, therefore, view matching should aim to produce as many correspondences...
//as there are features in the second frame, complicated because features must be reused for subsequent frames
//so might need to produce two sets of features for each image, for example, use a high and low threshold
//to determine which of the two sets a feature gets passed to, very good features: both, not so good features: between views
//what about refinement? use lower threshold to determine if gets refined, then at the end of refining step, add to lower threshold set (no test needed),
//and add to higher threshold set (if test passed)

//cannot only use lower threshold, because then matching time for between frame is far too long- is it actualy necessary though? between view can produce many correct correspondences,
//even with high detection threshold, as long as the matching threshold is low enough - lets leave this for now.

//idea for view matching algorithm: group views into bins divided by row, then all features within a bin get compared, because of deviation stuff, might need to include features in up
//to 2 bins (for features on edge of bin)
__global__ void matchViews() {

}

//problem: diffuclt to use additional optimizations (i.e. sign of laplacian, row, etc.), because an individual key point cannot skip its current candidate match...
//unless all of the other key points in its block group also skip that match.


//exhaustive search

//if a block of 64 threads is assigned to each key point, it becomes much easier to include these optimizations, although the number of global memory accesses...
//will also increase greatly
__global__ void matchFrames(float *d_D1, float *d_D2, int *d_m, float *d_H1, float *d_H2, int d_k1, int d_k2) {
	int l_k1 = threadIdx.y + blockIdx.x * 4;
	float l_e11, l_e12;
	if (l_k1 < d_k1) {
		l_e11 = d_D1[threadIdx.x + l_k1 * 64];
		l_e12 = d_D1[threadIdx.x + 32 + l_k1 * 64];
	}
	__shared__ float s_d[256];
	__shared__ float s_v[64];
	float l_bd = FLT_MAX, l_sbd = FLT_MAX;
	int l_bi = -1;
	for (int l_k2 = 0; l_k2 < d_k2; l_k2++) {
		int l_ei1 = threadIdx.y * 64;
		int l_ei2 = threadIdx.x + threadIdx.y * 32;
		if (l_ei2 < 64)
			s_v[l_ei2] = d_D2[l_ei2 + l_k2 * 64];
		__syncthreads();
		float l_d = l_e11 - s_v[threadIdx.x];
		s_d[threadIdx.x + l_ei1] = l_d * l_d;
		l_d = l_e12 - s_v[threadIdx.x + 32];
		s_d[threadIdx.x + 32 + l_ei1] = l_d * l_d;
		reduce(s_d + l_ei1);
		__syncthreads();
		l_d = sqrtf(s_d[l_ei1]);
		if (l_d < l_bd) {
			l_sbd = l_bd;
			l_bd = l_d;
			l_bi = l_k2;
		} else if (l_d < l_sbd)
			l_sbd = l_d;
	}
	if (threadIdx.x == 0 && l_k1 < d_k1 && l_bd < MATCHING_DISTANCE_THRESHOLD && l_bd / l_sbd < MATCHING_RATIO_THRESHOLD) {
		//key phrase: currently "array of structures" but "structure of arrays" needed for coalescing (previous version), though performance gain negligable in this case
		int l_n = atomicAdd(d_m, 1);
		d_H1[l_n + MAX_KEYPOINTS * 3] = l_k1;
		d_H2[l_n + MAX_KEYPOINTS * 3] = l_bi;
	}
}

//refer to as integral image/ or "summed area table"

//do both images within thread to potentially improve instruction level parallelism?

__global__ void sumVertically(int *d_U, unsigned char *d_I, int d_IP, int d_w, int d_h) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_c < d_w + 1) {
		unsigned int l_s = 0;
		d_U[l_c] = 0;
		if (l_c > 0)
			for (int l_r = 0; l_r < d_h; l_r++) {
				l_s += d_I[l_c - 1 + l_r * d_IP];
				d_U[l_c + (l_r + 1) * (d_w + 1)] = l_s;
			}
		else
			for (int l_r = 0; l_r < d_h; l_r++)
				d_U[(l_r + 1) * (d_w + 1)] = 0;
	}
}

__global__ void sumHorizontally(int *d_U, int d_w, int d_h) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_r < d_h + 1) {
		d_U[l_r * (d_w + 1)] = 0;
		if (l_r > 0)
			for (int l_c = 0; l_c < d_w; l_c++)
				d_U[l_c + 1 + l_r * (d_w + 1)] += d_U[l_c + l_r * (d_w + 1)];
		else
			for (int l_c = 0; l_c < d_w; l_c++)
				d_U[l_c + 1] = 0;
	}
}

__device__ void reduce(volatile float *s_d) {
	s_d[threadIdx.x] += s_d[threadIdx.x + 32];
	s_d[threadIdx.x] += s_d[threadIdx.x + 16];
	s_d[threadIdx.x] += s_d[threadIdx.x + 8];
	s_d[threadIdx.x] += s_d[threadIdx.x + 4];
	s_d[threadIdx.x] += s_d[threadIdx.x + 2];
	s_d[threadIdx.x] += s_d[threadIdx.x + 1];	
}

__device__ float filter(int *d_U, int d_w, int d_r, int d_c, int d_Lx, int d_Ly) {
	return (float) d_U[d_c + 1 + (d_r + 1) * (d_w + 1)] - d_U[d_c + 1 - d_Lx + (d_r + 1) * (d_w + 1)] - d_U[d_c + 1 + (d_r + 1 - d_Ly) * (d_w + 1)] + d_U[d_c + 1 - d_Lx + (d_r + 1 - d_Ly) * (d_w + 1)];
}

__device__ float dx(int *d_U, int d_w, int d_r, int d_c, int d_L) {
	return filter(d_U, d_w, d_r + d_L, d_c - 1, d_L, 2 * d_L + 1) - filter(d_U, d_w, d_r + d_L, d_c + d_L, d_L, 2 * d_L + 1);
}

__device__ float dy(int *d_U, int d_w, int d_r, int d_c, int d_L) {
	return filter(d_U, d_w, d_r - 1, d_c + d_L, 2 * d_L + 1, d_L) - filter(d_U, d_w, d_r + d_L, d_c + d_L, 2 * d_L + 1, d_L);
}

__device__ float dxx(int *d_U, int d_w, int d_r, int d_c, int d_L) {
	return filter(d_U, d_w, d_r + d_L, d_c + d_L + d_L / 2, 3 * d_L, 2 * d_L + 1) - 3 * filter(d_U, d_w, d_r + d_L, d_c + d_L / 2, d_L, 2 * d_L + 1);
}

__device__ float dyy(int *d_U, int d_w, int d_r, int d_c, int d_L) {
	return filter(d_U, d_w, d_r + d_L + d_L / 2, d_c + d_L, 2 * d_L + 1, 3 * d_L) - 3 * filter(d_U, d_w, d_r + d_L / 2, d_c + d_L, 2 * d_L + 1, d_L);
}

__device__ float dxy(int *d_U, int d_w, int d_r, int d_c, int d_L) {
	return filter(d_U, d_w, d_r + d_L, d_c - 1, d_L, d_L) + filter(d_U, d_w, d_r - 1, d_c + d_L, d_L, d_L) - filter(d_U, d_w, d_r + d_L, d_c + d_L, d_L, d_L) - filter(d_U, d_w, d_r - 1, d_c - 1, d_L, d_L);
}


void keyPoints() {
	PNG l("im2G.png"), re("im6G.png");
	unsigned char *d_I, *d_I2;
	int d_IP, d_IP2;
	int *d_U, *d_U2;
	float *d_H, *d_H2;
	float *d_D, *d_D2;
	int k = 0, r = 0, k2 = 0, r2 = 0;
	int *d_k, *d_r, *d_k2, *d_r2;
	float x[MAX_KEYPOINTS], y[MAX_KEYPOINTS], s[MAX_KEYPOINTS];

	//left
	cudaMallocPitch(&d_I, (size_t *) &d_IP, l.width * sizeof(unsigned char), l.height);
	cudaMalloc(&d_U, (l.width + 1) * (l.height + 1) * sizeof(int));
	cudaMalloc(&d_H, l.width * l.height * SCALES * sizeof(float));
	cudaMalloc(&d_k, sizeof(int));
	cudaMalloc(&d_r, sizeof(int));
	cudaMalloc(&d_D, MAX_KEYPOINTS * 64 * sizeof(float));

	cudaMemcpy2D(d_I, d_IP, l.data, l.width * sizeof(unsigned char), l.width * sizeof(unsigned char), l.height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, &r, sizeof(int), cudaMemcpyHostToDevice);

	//right
	cudaMallocPitch(&d_I2, (size_t *) &d_IP2, l.width * sizeof(unsigned char), l.height);
	cudaMalloc(&d_U2, (l.width + 1) * (l.height + 1) * sizeof(int));
	cudaMalloc(&d_H2, l.width * l.height * SCALES * sizeof(float));
	cudaMalloc(&d_k2, sizeof(int));
	cudaMalloc(&d_r2, sizeof(int));
	cudaMalloc(&d_D2, MAX_KEYPOINTS * 64 * sizeof(float));

	cudaMemcpy2D(d_I2, d_IP2, re.data, l.width * sizeof(unsigned char), l.width * sizeof(unsigned char), l.height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_k2, &k2, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r2, &r2, sizeof(int), cudaMemcpyHostToDevice);

	//left
	sumVertically<<<10, 192>>>(d_U, d_I, d_IP, l.width, l.height);
	sumHorizontally<<<8, 192>>>(d_U, l.width, l.height);
	compute<<<28125, 96>>>(d_H, d_U, l.width, l.height);
	supress<<<21094, 128>>>(d_k, d_H, l.width, l.height, d_I, d_IP);
	cudaMemcpy(&k, d_k, 4, cudaMemcpyDeviceToHost);
	k = std::min(k, MAX_KEYPOINTS);
	refine<<<(k + 191) / 192, 192>>>(d_H, d_r, l.width, l.height, k);
	cudaMemcpy(&r, d_r, sizeof(int), cudaMemcpyDeviceToHost);
	describe<<<r, dim3(5, 5, 16)>>>(d_D, d_H, d_U, l.width);
	normalize<<<r, 64>>>(d_D, d_H);
	cudaMemcpy(x, d_H, k * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_H + MAX_KEYPOINTS, k * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(s, d_H + MAX_KEYPOINTS * 2, k * sizeof(float), cudaMemcpyDeviceToHost);

	//right
	sumVertically<<<10, 192>>>(d_U2, d_I2, d_IP2, l.width, l.height);
	sumHorizontally<<<8, 192>>>(d_U2, l.width, l.height);
	compute<<<28125, 96>>>(d_H2, d_U2, l.width, l.height);//change to 128 size blocks?
	supress<<<21094, 128>>>(d_k2, d_H2, l.width, l.height, d_I2, d_IP2);
	cudaMemcpy(&k2, d_k2, 4, cudaMemcpyDeviceToHost);
	k2 = std::min(k2, MAX_KEYPOINTS);
	refine<<<(k2 + 191) / 192, 192>>>(d_H2, d_r2, l.width, l.height, k2);
	cudaMemcpy(&r2, d_r2, sizeof(int), cudaMemcpyDeviceToHost);
	describe<<<r2, dim3(5, 5, 16)>>>(d_D2, d_H2, d_U2, l.width);
	normalize<<<r2, 64>>>(d_D2, d_H2);

	int m = 0;
	int *d_m;

	cudaMalloc(&d_m, sizeof(int));
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);

	matchFrames<<<(r + 3) / 4, dim3(32, 4)>>>(d_D, d_D2, d_m, d_H, d_H2, r, r2);
	cudaMemcpy(&m, d_m, sizeof(int), cudaMemcpyDeviceToHost);

	float matches[2000];
	float matches2[2000];

	cudaMemcpy(matches, d_H + 3 * MAX_KEYPOINTS, m * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matches2, d_H2 + 3 * MAX_KEYPOINTS, m * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m; i++)
		std::cout << matches[i] << "  " << matches2[i] << "\n\n";
	std::cout << m << "\n";
	std::cout << "KEYPOINTS(LEFT): " << r << "\n";
	std::cout << "KEYPOINTS(RIGHT): " << r2 << "\n";

	//(619l) - (684r) - (86m)
}