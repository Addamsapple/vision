#include <cuda_runtime.h>
#include <fstream>

#include "cam.hpp"
#include "tra.hpp"
#include "pos.hpp"
#include "pre.cuh"

#include "../capturing/jpg.hpp";

//#include <iostream>

#define EXTRA_REFINEMENT_ITERATIONS 0

#define COMPUTATION_THREADS 128
#define COMPUTATION_BLOCKS ((RECTIFIED_IMAGE_HEIGHT + COMPUTATION_THREADS - 1) / COMPUTATION_THREADS * 2)

#define MAXIMUM_DISPARITY UCHAR_MAX
#define OCCLUSION_COST 200.0f//300.0f

#define MATCH_TRANSITION 1
#define LEFT_OCCLUSION_TRANSITION 2
#define RIGHT_OCCLUSION_TRANSITION 3

#define RECONSTRUCTION_BLOCKS ((RECTIFIED_IMAGE_HEIGHT + COMPUTATION_THREADS - 1) / COMPUTATION_THREADS)

#define PROPAGATION_THREADS 128
#define PROPAGATION_THREAD_PIXELS 16
#define PROPAGATION_BLOCKS (((RECTIFIED_IMAGE_WIDTH - 2 + PROPAGATION_THREAD_PIXELS - 1) / (PROPAGATION_THREAD_PIXELS) * (RECTIFIED_IMAGE_HEIGHT - 2) + PROPAGATION_THREADS - 1) / PROPAGATION_THREADS)

unsigned char *d_tt;
size_t d_ttp;

float *d_ct;
size_t d_ctp;

unsigned char *dm;

template<int w, int h>
__global__ void computeSolutions(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp);
template<int w, int h>
__global__ void reconstructBestSolution(unsigned char *d_b, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp);
template<int w, int h>
__global__ void propagateDiscontinuities(unsigned char *d_b2, unsigned char *d_b1);
template<int w, int h>
__global__ void propagateOcclusions(unsigned char *d_b2, unsigned char *d_b1);

template<int w, int h>
__device__ void computeSolutionsRightwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp);
template<int w, int h>
__device__ void computeSolutionsLeftwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp);

void initializeDisparityMapComputation() {
	cudaMallocPitch(&d_tt, &d_ttp, RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), RECTIFIED_IMAGE_WIDTH * (MAXIMUM_DISPARITY + 1));
	d_ttp /= sizeof(unsigned char);
	cudaMallocPitch(&d_ct, &d_ctp, RECTIFIED_IMAGE_HEIGHT * sizeof(float), (MAXIMUM_DISPARITY + 1) * 4);
	d_ctp /= sizeof(float);
	dm = new unsigned char[RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT];
}

void computeDisparityMap() {
	computeSolutions<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<COMPUTATION_BLOCKS, COMPUTATION_THREADS>>>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
	reconstructBestSolution<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECONSTRUCTION_BLOCKS, COMPUTATION_THREADS>>>(d_ltri, d_tt, d_ct, d_ttp, d_ctp);
}

void refineDisparityMap() {
	unsigned char *buffers[2] = {d_ltri, d_rtri};
	propagateDiscontinuities<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<PROPAGATION_BLOCKS, PROPAGATION_THREADS>>>(buffers[1], buffers[0]);
	int bufferIndex = 1;
	for (int iteration = 0; iteration < EXTRA_REFINEMENT_ITERATIONS; iteration++) {
		propagateOcclusions<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<PROPAGATION_BLOCKS, PROPAGATION_THREADS>>>(buffers[1 - bufferIndex], buffers[bufferIndex]);
		bufferIndex = 1 - bufferIndex;
	}
	cudaMemcpy(dm, buffers[bufferIndex], RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void constructPointCloud() {
	std::ofstream p("points.p", std::ofstream::binary | std::ofstream::app);
	std::ofstream c("colours.p", std::ofstream::binary | std::ofstream::app);
	float point[3];
	float point2[3];
	unsigned char col[3] = {100, 100, 100};
	for (int row = 0; row < RECTIFIED_IMAGE_HEIGHT; row++)
		for (int column = 0; column < RECTIFIED_IMAGE_WIDTH; column++) {
			if (dm[row + column * RECTIFIED_IMAGE_HEIGHT] != 0) {
				point[0] = (column + LEFT_RECTIFIED_COLUMN - HORIZONTAL_PRINCIPAL_POINT) * BASELINE / dm[row + column * RECTIFIED_IMAGE_HEIGHT];
				point[1] = (row + TOP_RECTIFIED_ROW - VERTICAL_PRINCIPAL_POINT) * BASELINE / dm[row + column * RECTIFIED_IMAGE_HEIGHT];
				point[2] = HORIZONTAL_FOCAL_LENGTH * BASELINE / dm[row + column * RECTIFIED_IMAGE_HEIGHT];
				for (int dimension = 0; dimension < 3; dimension++)
					point2[dimension] = vectorDotProduct(point, orientation + (long long) dimension * 3, 3);//matrixInnerProduct(point, orientation + dimension, 3);
				combineVectors(point, point2, position, 1.0f, 1.0f, 3);
				point[0] *= -1.0f;
				point[1] *= -1.0f;
				p.write((char *) point, sizeof(float) * 3);


					float l_ud = (column + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_20 + (row + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_21 + LEFT_HOMOGRAPHY_22;
					float l_ux = ((column + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_00 + (row + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_01 + LEFT_HOMOGRAPHY_02) / l_ud;
					float l_uy = ((column + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_10 + (row + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_11 + LEFT_HOMOGRAPHY_12) / l_ud;
					float l_sr = l_ux * l_ux + l_uy * l_uy;
					float l_df = l_sr * l_sr * SECOND_LEFT_DISTORTION_COEFFICIENT + l_sr * FIRST_LEFT_DISTORTION_COEFFICIENT + 1.0;
					int l_dc = l_ux * l_df * LEFT_HORIZONTAL_FOCAL_LENGTH + LEFT_HORIZONTAL_PRINCIPAL_POINT;
					l_dc = std::min(DISTORTED_IMAGE_WIDTH - 1, std::max(0, l_dc));
					int l_dr = l_uy * l_df * LEFT_VERTICAL_FOCAL_LENGTH + LEFT_VERTICAL_PRINCIPAL_POINT;
					l_dr = std::min(DISTORTED_IMAGE_HEIGHT - 1, std::max(0, l_dr));

					c.write((char *) leftJPG + l_dc * 3 + l_dr * DISTORTED_IMAGE_WIDTH * 3, sizeof(unsigned char) * 3);





				//c.write((char *) leftJPG + column * 3 + row * DISTO, sizeof(unsigned char) * 3);
			}
		}
	p.close();
	c.close();
}

template<int w, int h>
__global__ void computeSolutions(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp) {
	if (blockIdx.x < gridDim.x / 2)
		computeSolutionsRightwards<w, h>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
	else
		computeSolutionsLeftwards<w, h>(d_ltri, d_rtri, d_tt, d_ct, d_ttp, d_ctp);
}

#define tt(row, column, offset) d_tt[row + d_ttp * ((column) * (MAXIMUM_DISPARITY + 1) + offset)]
#define ct(row, column, offset) d_ct[row + d_ctp * ((column) * (MAXIMUM_DISPARITY + 1) + offset)]

template<int w, int h>
__global__ void reconstructBestSolution(unsigned char *d_b, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_r < h) {
		float l_mc = FLT_MAX;
		int l_mi = -1;
		for (int l_do = 0; l_do < MAXIMUM_DISPARITY + 1; l_do++) {
			float l_c = ct(l_r, (w / 2 - 1) % 2, l_do) + ct(l_r, (w - 1) / 2 % 2 + 2, l_do);
			if (l_c < l_mc) {
				l_mc = l_c;
				l_mi = l_do;
			}		
		}
		int l_co = w / 2 - 1;
		unsigned char l_do = l_mi;
		while (l_co > -1) {
			unsigned char l_t = tt(l_r, l_co, l_do);
			if (l_t == MATCH_TRANSITION) {
				d_b[l_r + l_co * h] = MAXIMUM_DISPARITY - l_do;
				l_co--;
			} else if (l_t == LEFT_OCCLUSION_TRANSITION) {
				d_b[l_r + l_co * h] = 0;
				l_co--;
				l_do++;
			} else
				l_do--;
		}
		l_co = w / 2;
		l_do = l_mi;
		while (l_co < w) {
			unsigned char l_map = tt(l_r, l_co, l_do);
			if (l_map == MATCH_TRANSITION) {
				d_b[l_r + l_co * h] = MAXIMUM_DISPARITY - l_do;
				l_co++;
			} else if (l_map == LEFT_OCCLUSION_TRANSITION) {
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
__device__ void computeSolutionsRightwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_r < h) {
		for (int l_do = 0; l_do < MAXIMUM_DISPARITY; l_do++) {
			ct(l_r, 0, l_do) = OCCLUSION_COST;
			tt(l_r, 0, l_do) = LEFT_OCCLUSION_TRANSITION;
		}
		float l_lp = d_ltri[l_r];
		float l_rp = d_rtri[l_r];
		float l_mc = (l_lp - l_rp) * (l_lp - l_rp);
		float l_roc = OCCLUSION_COST * 2;
		float l_c = fminf(l_mc, l_roc);
		int l_i = 0;
		ct(l_r, l_i, MAXIMUM_DISPARITY) = l_c;
		if (l_c == l_mc)
			tt(l_r, 0, MAXIMUM_DISPARITY) = MATCH_TRANSITION;
		else
			tt(l_r, 0, MAXIMUM_DISPARITY) = RIGHT_OCCLUSION_TRANSITION;
		float l_loc;
		int l_do;
		for (int l_co = 1; l_co < w / 2; l_co++) {
			l_i = 1 - l_i;
			l_lp = d_ltri[l_r + l_co * h];
			if (l_co < MAXIMUM_DISPARITY) {
				for (int l_offset = 0; l_offset < MAXIMUM_DISPARITY - l_co; l_offset++) {
					ct(l_r, l_i, l_offset) = ct(l_r, 1 - l_i, l_offset + 1) + OCCLUSION_COST;
					tt(l_r, l_co, l_offset) = LEFT_OCCLUSION_TRANSITION;
				}
				l_do = MAXIMUM_DISPARITY - l_co;
			} else {
				l_rp = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY) * h];
				l_mc = ct(l_r, 1 - l_i, 0) + (l_lp - l_rp) * (l_lp - l_rp);
				l_loc = ct(l_r, 1 - l_i, 1) + OCCLUSION_COST;
				l_c = fminf(l_mc, l_loc);
				ct(l_r, l_i, 0) = l_c;
				if (l_c == l_mc)
					tt(l_r, l_co, 0) = MATCH_TRANSITION;
				else
					tt(l_r, l_co, 0) = LEFT_OCCLUSION_TRANSITION;
				l_do = 1;
			}
			for (; l_do < MAXIMUM_DISPARITY; l_do++) {
				l_rp = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY + l_do) * h];
				l_mc = ct(l_r, 1 - l_i, l_do) + (l_lp - l_rp) * (l_lp - l_rp);
				l_loc = ct(l_r, 1 - l_i, l_do + 1) + OCCLUSION_COST;
				l_roc = ct(l_r, l_i, l_do - 1) + OCCLUSION_COST;
				l_c = fminf(fminf(l_mc, l_loc), l_roc);
				ct(l_r, l_i, l_do) = l_c;
				if (l_c == l_mc)
					tt(l_r, l_co, l_do) = MATCH_TRANSITION;
				else if (l_c == l_roc)
					tt(l_r, l_co, l_do) = RIGHT_OCCLUSION_TRANSITION;
				else
					tt(l_r, l_co, l_do) = LEFT_OCCLUSION_TRANSITION;
			}
			l_rp = d_rtri[l_r + l_co * h];
			l_mc = ct(l_r, 1 - l_i, MAXIMUM_DISPARITY) + (l_lp - l_rp) * (l_lp - l_rp);
			l_roc = ct(l_r, l_i, MAXIMUM_DISPARITY - 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_roc);
			ct(l_r, l_i, MAXIMUM_DISPARITY) = l_c;
			if (l_c == l_mc)
				tt(l_r, l_co, MAXIMUM_DISPARITY) = MATCH_TRANSITION;
			else
				tt(l_r, l_co, MAXIMUM_DISPARITY) = RIGHT_OCCLUSION_TRANSITION;
		}
	}
}

template<int w, int h>
__device__ void computeSolutionsLeftwards(unsigned char *d_ltri, unsigned char *d_rtri, unsigned char *d_tt, float *d_ct, size_t d_ttp, size_t d_ctp) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x - gridDim.x * blockDim.x / 2;
	if (l_r < h) {
		float l_li = d_ltri[l_r + (w - 1) * h];
		float l_ri = d_rtri[l_r + (w - 1) * h];
		float l_mc = (l_li - l_ri) * (l_li - l_ri);
		float l_lc = OCCLUSION_COST * 2;
		float l_c = fminf(l_mc, l_lc);
		int l_i = 0;
		ct(l_r, l_i + 2, MAXIMUM_DISPARITY) = l_c;
		if (l_c == l_mc)
			tt(l_r, w - 1, MAXIMUM_DISPARITY) = MATCH_TRANSITION;
		else
			tt(l_r, w - 1, MAXIMUM_DISPARITY) = LEFT_OCCLUSION_TRANSITION;
		float l_rc;
		for (int l_do = MAXIMUM_DISPARITY - 1; l_do > -1; l_do--) {
			l_ri = d_rtri[l_r + (w - 1 - MAXIMUM_DISPARITY + l_do) * h];
			l_mc = OCCLUSION_COST + (l_li - l_ri) * (l_li - l_ri);
			l_rc = ct(l_r, l_i + 2, l_do + 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_rc);
			ct(l_r, l_i + 2, l_do) = l_c;
			if (l_c == l_mc)
				tt(l_r, w - 1, l_do) = MATCH_TRANSITION;
			else
				tt(l_r, w - 1, l_do) = RIGHT_OCCLUSION_TRANSITION;
		}
		for (int l_co = w - 2; l_co > w / 2 - 1; l_co--) {
			l_i = 1 - l_i;
			l_li = d_ltri[l_r + l_co * h];
			l_ri = d_rtri[l_r + l_co * h];
			l_mc = ct(l_r, 3 - l_i, MAXIMUM_DISPARITY) + (l_li - l_ri) * (l_li - l_ri);
			l_lc = ct(l_r, 3 - l_i, MAXIMUM_DISPARITY - 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_lc);
			ct(l_r, l_i + 2, MAXIMUM_DISPARITY) = l_c;
			if (l_c == l_mc)
				tt(l_r, l_co, MAXIMUM_DISPARITY) = MATCH_TRANSITION;
			else
				tt(l_r, l_co, MAXIMUM_DISPARITY) = LEFT_OCCLUSION_TRANSITION;
			for (int l_do = MAXIMUM_DISPARITY - 1; l_do > 0; l_do--) {
				l_ri = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY + l_do) * h];
				l_mc = ct(l_r, 3 - l_i, l_do) + (l_li - l_ri) * (l_li - l_ri);
				l_lc = ct(l_r, 3 - l_i, l_do - 1) + OCCLUSION_COST;
				l_rc = ct(l_r, l_i + 2, l_do + 1) + OCCLUSION_COST;
				l_c = fminf(fminf(l_mc, l_lc), l_rc);
				ct(l_r, l_i + 2, l_do) = l_c;
				if (l_c == l_mc)
					tt(l_r, l_co, l_do) = MATCH_TRANSITION;
				else if (l_c == l_rc)
					tt(l_r, l_co, l_do) = RIGHT_OCCLUSION_TRANSITION;
				else
					tt(l_r, l_co, l_do) = LEFT_OCCLUSION_TRANSITION;

			}
			l_ri = d_rtri[l_r + (l_co - MAXIMUM_DISPARITY) * h];
			l_mc = ct(l_r, 3 - l_i, 0) + (l_li - l_ri) * (l_li - l_ri);
			l_rc = ct(l_r, l_i + 2, 1) + OCCLUSION_COST;
			l_c = fminf(l_mc, l_rc);
			ct(l_r, l_i + 2, 0) = l_c;
			if (l_c == l_mc)
				tt(l_r, l_co, 0) = MATCH_TRANSITION;
			else
				tt(l_r, l_co, 0) = RIGHT_OCCLUSION_TRANSITION;
		}
	}
}