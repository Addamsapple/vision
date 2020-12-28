#include <cuda_runtime.h>

#include "cam.hpp"
#include "png.hpp"
#include "pre.cuh"

#define MAXIMUM_DISPARITY UCHAR_MAX
#define OCCLUSION_COST 300.0f

#define MATCH 1
#define LEFT_OCCLUSION 2
#define RIGHT_OCCLUSION 3

#define COMPUTATION_THREADS 128
#define COMPUTATION_BLOCKS ((RECTIFIED_IMAGE_HEIGHT + COMPUTATION_THREADS - 1) / COMPUTATION_THREADS * 2)
#define RECONSTRUCTION_BLOCKS ((RECTIFIED_IMAGE_HEIGHT + COMPUTATION_THREADS - 1) / COMPUTATION_THREADS)

#define PROPAGATION_THREADS 128
#define PROPAGATION_THREAD_PIXELS 16
#define PROPAGATION_BLOCKS (((RECTIFIED_IMAGE_WIDTH - 2 + PROPAGATION_THREAD_PIXELS - 1) / (PROPAGATION_THREAD_PIXELS) * (RECTIFIED_IMAGE_HEIGHT - 2) + PROPAGATION_THREADS - 1) / PROPAGATION_THREADS)

unsigned char *d_tt;
size_t d_ttp;

float *d_ct;
size_t d_ctp;

template<int w, int h>
__global__ void computeSolutions(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp);
template<int w, int h>
__global__ void reconstructSolution(unsigned char *d_b, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp);
template<int w, int h>
__global__ void propagateDiscontinuities(unsigned char *d_b2, unsigned char *d_b1);
template<int w, int h>
__global__ void propagateOcclusions(unsigned char *d_b2, unsigned char *d_b1);

template<int w, int h>
__device__ void computeSolutionsRightwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp);
template<int w, int h>
__device__ void computeSolutionsLeftwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp);

void initializeDisparityMapComputation() {
	cudaMallocPitch(&d_tt, &d_ttp, RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), RECTIFIED_IMAGE_WIDTH * (MAXIMUM_DISPARITY + 1));
	d_ttp /= sizeof(unsigned char);
	cudaMallocPitch(&d_ct, &d_ctp, RECTIFIED_IMAGE_HEIGHT * sizeof(float), (MAXIMUM_DISPARITY + 1) * 4);
	d_ctp /= sizeof(float);
}

void computeDisparityMap() {
	computeSolutions<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<COMPUTATION_BLOCKS, COMPUTATION_THREADS>>>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
	reconstructSolution<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECONSTRUCTION_BLOCKS, COMPUTATION_THREADS>>>(d_ltri, d_tt, d_ct, d_ttp, d_ctp);
}

void refineDisparityMap(unsigned char *a, int iterations) {
	unsigned char *buffers[2] = {d_ltri, d_rtri};
	int index = 0;
	if (iterations > 0) {
		propagateDiscontinuities<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<PROPAGATION_BLOCKS, PROPAGATION_THREADS>>>(buffers[1 - index], buffers[index]);
		index = 1 - index;
		for (int iteration = 1; iteration < iterations; iteration++) {
			propagateOcclusions<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<PROPAGATION_BLOCKS, PROPAGATION_THREADS>>>(buffers[1 - index], buffers[index]);
			index = 1 - index;
		}
	}
	cudaMemcpy(a, buffers[index], RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

template<int w, int h>
__global__ void computeSolutions(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp) {
	if (blockIdx.x < gridDim.x / 2)
		computeSolutionsRightwards<w, h>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
	else
		computeSolutionsLeftwards<w, h>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
}

//rename these
#define cost(row, column, offset) d_ct[row + d_ctp * ((column) * (MAXIMUM_DISPARITY + 1) + offset)]
#define map(row, column, offset) d_tt[row + d_ttp * ((column) * (MAXIMUM_DISPARITY + 1) + offset)]

template<int w, int h>
__global__ void reconstructSolution(unsigned char *d_b, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_r < h) {
		float l_mc = FLT_MAX;
		int l_mi = -1;
		for (int l_do = 0; l_do < MAXIMUM_DISPARITY + 1; l_do++) {
			float l_c = cost(l_r, (w / 2 - 1) % 2, l_do) + cost(l_r, (w - 1) / 2 % 2 + 2, l_do);
			if (l_c < l_mc) {
				l_mc = l_c;
				l_mi = l_do;
			}		
		}
		int l_co = w / 2 - 1;
		unsigned char l_do = l_mi;
		while (l_co > -1) {
			unsigned char l_t = map(l_r, l_co, l_do);
			if (l_t == MATCH) {
				d_b[l_r + l_co * h] = MAXIMUM_DISPARITY - l_do;
				l_co--;
			} else if (l_t == LEFT_OCCLUSION) {
				d_b[l_r + l_co * h] = 0;
				l_co--;
				l_do++;
			} else
				l_do--;
		}
		//test
		l_co = w / 2;
		l_do = l_mi;
		while (l_co < w) {
			unsigned char l_map = map(l_r, l_co, l_do);
			if (l_map == MATCH) {
				d_b[l_r + l_co * h] = MAXIMUM_DISPARITY - l_do;
				l_co++;
			} else if (l_map == LEFT_OCCLUSION) {
				d_b[l_r + l_co * h] = 0;
				l_co++;
				l_do--;
			} else
				l_do++;
		}
	}
}

template<int w, int h>
__global__ void propagateDiscontinuities(unsigned char *d_b2, unsigned char *d_b1) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	int l_c = l_r / (h - 2);
	l_r -= l_c * (h - 2) - 1;
	l_c = l_c * PROPAGATION_THREAD_PIXELS + 1;
	for (int l_co = 0; l_co < PROPAGATION_THREAD_PIXELS && l_c + l_co < w - 1; l_co++) {
		int l_ld = d_b1[l_r + (l_c + l_co - 1) * h];
		int l_rd = d_b1[l_r + (l_c + l_co + 1) * h];
		int l_ud = d_b1[l_r - 1 + (l_c + l_co) * h];
		int l_dd = d_b1[l_r + 1 + (l_c + l_co) * h];
		if (l_ld == 0 || l_ld != l_rd || l_ld != l_ud || l_ld != l_dd)
			d_b2[l_r + (l_c + l_co) * h] = 0;
		else {
			int l_d = d_b1[l_r + (l_c + l_co) * h];
			if (l_d != l_ld)
				d_b2[l_r + (l_c + l_co) * h] = 0;
			else
				d_b2[l_r + (l_c + l_co) * h] = l_d;
		}
	}
}

//does not invalidate border pixels
template<int w, int h>
__global__ void propagateOcclusions(unsigned char *d_b2, unsigned char *d_b1) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	int l_c = l_r / (h - 2);
	l_r -= l_c * (h - 2) - 1;
	l_c = l_c * PROPAGATION_THREAD_PIXELS + 1;
	for (int l_co = 0; l_co < PROPAGATION_THREAD_PIXELS && l_c + l_co < w - 1; l_co++) {
		int l_ld = d_b1[l_r + (l_c + l_co - 1) * h];
		int l_rd = d_b1[l_r + (l_c + l_co + 1) * h];
		int l_ud = d_b1[l_r - 1 + (l_c + l_co) * h];
		int l_dd = d_b1[l_r + 1 + (l_c + l_co) * h];
		if (l_ld == 0 || l_rd == 0 || l_ud == 0 || l_dd == 0)
			d_b2[l_r + (l_c + l_co) * h] = 0;
		else
			d_b2[l_r + (l_c + l_co) * h] = d_b1[l_r + (l_c + l_co) * h];
	}
}

template<int w, int h>
__device__ void computeSolutionsRightwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_r < h) {
		for (int l_do = 0; l_do < MAXIMUM_DISPARITY; l_do++) {
			cost(l_r, 0, l_do) = OCCLUSION_COST;
			map(l_r, 0, l_do) = LEFT_OCCLUSION;
		}
		float l_lp = d_ltri[l_r];
		float l_rp = d_rtri[l_r];
		float l_mc = (l_lp - l_rp) * (l_lp - l_rp);
		float l_roc = OCCLUSION_COST * 2;
		float l_c = fminf(l_mc, l_roc);
		int l_i = 0;
		cost(l_r, l_i, MAXIMUM_DISPARITY) = l_c;
		if (l_c == l_mc)
			map(l_r, 0, MAXIMUM_DISPARITY) = MATCH;
		else
			map(l_r, 0, MAXIMUM_DISPARITY) = RIGHT_OCCLUSION;
		float l_loc;
		int l_do;
		for (int l_co = 1; l_co < w / 2; l_co++) {
			l_i = 1 - l_i;
			l_lp = d_ltri[l_r + l_co * h];
			if (l_co < MAXIMUM_DISPARITY) {
				for (int l_offset = 0; l_offset < MAXIMUM_DISPARITY - l_co; l_offset++) {
					cost(l_r, l_i, l_offset) = cost(l_r, 1 - l_i, l_offset + 1) + OCCLUSION_COST;
					map(l_r, l_co, l_offset) = LEFT_OCCLUSION;
				}
				l_do = MAXIMUM_DISPARITY - l_co;
			} else {
				l_rp = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY) * h];
				l_mc = cost(l_r, 1 - l_i, 0) + (l_lp - l_rp) * (l_lp - l_rp);
				l_loc = cost(l_r, 1 - l_i, 1) + OCCLUSION_COST;
				l_c = fminf(l_mc, l_loc);
				cost(l_r, l_i, 0) = l_c;
				if (l_c == l_mc)
					map(l_r, l_co, 0) = MATCH;
				else
					map(l_r, l_co, 0) = LEFT_OCCLUSION;
				l_do = 1;
			}
			for (; l_do < MAXIMUM_DISPARITY; l_do++) {
				l_rp = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY + l_do) * h];
				l_mc = cost(l_r, 1 - l_i, l_do) + (l_lp - l_rp) * (l_lp - l_rp);
				l_loc = cost(l_r, 1 - l_i, l_do + 1) + OCCLUSION_COST;
				l_roc = cost(l_r, l_i, l_do - 1) + OCCLUSION_COST;
				l_c = fminf(fminf(l_mc, l_loc), l_roc);
				cost(l_r, l_i, l_do) = l_c;
				if (l_c == l_mc)
					map(l_r, l_co, l_do) = MATCH;
				else if (l_c == l_roc)
					map(l_r, l_co, l_do) = RIGHT_OCCLUSION;
				else
					map(l_r, l_co, l_do) = LEFT_OCCLUSION;
			}
			l_rp = d_rtri[l_r + l_co * h];
			l_mc = cost(l_r, 1 - l_i, MAXIMUM_DISPARITY) + (l_lp - l_rp) * (l_lp - l_rp);
			l_roc = cost(l_r, l_i, MAXIMUM_DISPARITY - 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_roc);
			cost(l_r, l_i, MAXIMUM_DISPARITY) = l_c;
			if (l_c == l_mc)
				map(l_r, l_co, MAXIMUM_DISPARITY) = MATCH;
			else
				map(l_r, l_co, MAXIMUM_DISPARITY) = RIGHT_OCCLUSION;
		}
	}
}

template<int w, int h>
__device__ void computeSolutionsLeftwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, int d_ttp, int d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x - gridDim.x * blockDim.x / 2;
	if (l_r < h) {
		float l_li = d_ltri[l_r + (w - 1) * h];
		float l_ri = d_rtri[l_r + (w - 1) * h];
		float l_mc = (l_li - l_ri) * (l_li - l_ri);
		float l_lc = OCCLUSION_COST * 2;
		float l_c = fminf(l_mc, l_lc);
		int l_i = 0;
		cost(l_r, l_i + 2, MAXIMUM_DISPARITY) = l_c;
		if (l_c == l_mc)
			map(l_r, w - 1, MAXIMUM_DISPARITY) = MATCH;
		else
			map(l_r, w - 1, MAXIMUM_DISPARITY) = LEFT_OCCLUSION;
		float l_rc;
		for (int l_do = MAXIMUM_DISPARITY - 1; l_do > -1; l_do--) {
			l_ri = d_rtri[l_r + (w - 1 - MAXIMUM_DISPARITY + l_do) * h];
			l_mc = OCCLUSION_COST + (l_li - l_ri) * (l_li - l_ri);
			l_rc = cost(l_r, l_i + 2, l_do + 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_rc);
			cost(l_r, l_i + 2, l_do) = l_c;
			if (l_c == l_mc)
				map(l_r, w - 1, l_do) = MATCH;
			else
				map(l_r, w - 1, l_do) = RIGHT_OCCLUSION;
		}
		for (int l_co = w - 2; l_co > w / 2 - 1; l_co--) {
			l_i = 1 - l_i;
			l_li = d_ltri[l_r + l_co * h];
			l_ri = d_rtri[l_r + l_co * h];
			l_mc = cost(l_r, 3 - l_i, MAXIMUM_DISPARITY) + (l_li - l_ri) * (l_li - l_ri);
			l_lc = cost(l_r, 3 - l_i, MAXIMUM_DISPARITY - 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_lc);
			cost(l_r, l_i + 2, MAXIMUM_DISPARITY) = l_c;
			if (l_c == l_mc)
				map(l_r, l_co, MAXIMUM_DISPARITY) = MATCH;
			else
				map(l_r, l_co, MAXIMUM_DISPARITY) = LEFT_OCCLUSION;
			for (int l_do = MAXIMUM_DISPARITY - 1; l_do > 0; l_do--) {
				l_ri = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY + l_do) * h];
				l_mc = cost(l_r, 3 - l_i, l_do) + (l_li - l_ri) * (l_li - l_ri);
				l_lc = cost(l_r, 3 - l_i, l_do - 1) + OCCLUSION_COST;
				l_rc = cost(l_r, l_i + 2, l_do + 1) + OCCLUSION_COST;
				l_c = fminf(fminf(l_mc, l_lc), l_rc);
				cost(l_r, l_i + 2, l_do) = l_c;
				if (l_c == l_mc)
					map(l_r, l_co, l_do) = MATCH;
				else if (l_c == l_rc)
					map(l_r, l_co, l_do) = RIGHT_OCCLUSION;
				else
					map(l_r, l_co, l_do) = LEFT_OCCLUSION;

			}
			l_ri = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY) * h];
			l_mc = cost(l_r, 3 - l_i, 0) + (l_li - l_ri) * (l_li - l_ri);
			l_rc = cost(l_r, l_i + 2, 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_rc);
			cost(l_r, l_i + 2, 0) = l_c;
			if (l_c == l_mc)
				map(l_r, l_co, 0) = MATCH;
			else
				map(l_r, l_co, 0) = RIGHT_OCCLUSION;
		}
	}
}