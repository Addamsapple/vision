#include <algorithm>
#include <cuda_runtime.h>

#include "cam.hpp"
#include "png.hpp"
#include "pre.cuh"

#define COMPUTATION_THREADS 96
#define COMPUTATION_BLOCKS ((RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT + COMPUTATION_THREADS - 1) / COMPUTATION_THREADS)

#define COMPUTATION_SCALES 10
#define COMPUTATION_PADDING 245

#define MAXIMUM_INTEREST_POINTS 1000

#define SUPPRESSION_THREADS 128
#define SUPPRESSION_BLOCKS ((RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT + SUPPRESSION_THREADS - 1) / SUPPRESSION_THREADS)

#define SUPRESSION_THRESHOLD 10000.0f

#define REFINEMENT_THREADS 192

#define REFINEMENT_EPSILON 0.00001f

#define DESCRIPTOR_BLOCK_DIMENSIONS 4
#define DESCRIPTOR_BLOCK_SIZE 5
#define DESCRIPTOR_GRID_SIZE 4

#define DESCRIPTOR_DIMENSIONS (DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE * DESCRIPTOR_BLOCK_DIMENSIONS)
#define DESCRIPTOR_WIDTH 10
#define DESCRIPTOR_SIGMA 0.4f

#define MATCHING_THREAD_INTEREST_POINTS 4

#define MATCHING_DISTANCE_THRESHOLD 0.1f
#define MATCHING_DISTANCE_RATIO_THRESHOLD 0.5f

float *d_lhd;
float *d_rhd;

int *d_lip;
int *d_lipc;
int lipc;

int *d_rip;
int *d_ripc;
int ripc;

float *d_lrip;
int *d_lripc;
int lripc;

float *d_rrip;
int *d_rripc;
int rripc;

float *d_lipd;
float *d_ripd;

int *d_ipm;
int *d_ipmc;
int ipmc;

__constant__ int d_css[COMPUTATION_SCALES] = {1, 1, 1, 1, 2, 2, 4, 4, 8, 8};
__constant__ int d_sss[COMPUTATION_SCALES] = {1, 1, 1, 2, 2, 4, 4, 8, 8, 8};
__constant__ int d_sso[COMPUTATION_SCALES * 3] = {0, 0, 0, -1, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, -2, 0, 1, -1, 0, 1, 0, 0, 0};
__constant__ int d_l[COMPUTATION_SCALES] = {3, 5, 7, 9, 13, 17, 25, 33, 49, 65};

__constant__ float d_gk[12][12] = {
	0.014614763f,	0.013958917f,	0.012162744f,	0.00966788f,	0.00701053f,	0.004637568f,	0.002798657f,	0.001540738f,	0.000773799f,	0.000354525f,	0.000148179f,	0.0f,
	0.013958917f,	0.013332502f,	0.011616933f,	0.009234028f,	0.006695928f,	0.004429455f,	0.002673066f,	0.001471597f,	0.000739074f,	0.000338616f,	0.000141529f,	0.0f,
	0.012162744f,	0.011616933f,	0.010122116f,	0.008045833f,	0.005834325f,	0.003859491f,	0.002329107f,	0.001282238f,	0.000643973f,	0.000295044f,	0.000123318f,	0.0f,
	0.00966788f,	0.009234028f,	0.008045833f,	0.006395444f,	0.004637568f,	0.003067819f,	0.001851353f,	0.001019221f,	0.000511879f,	0.000234524f,	9.80224E-05f,	0.0f,
	0.00701053f,	0.006695928f,	0.005834325f,	0.004637568f,	0.003362869f,	0.002224587f,	0.001342483f,	0.000739074f,	0.000371182f,	0.000170062f,	7.10796E-05f,	0.0f,
	0.004637568f,	0.004429455f,	0.003859491f,	0.003067819f,	0.002224587f,	0.001471597f,	0.000888072f,	0.000488908f,	0.000245542f,	0.000112498f,	4.70202E-05f,	0.0f,
	0.002798657f,	0.002673066f,	0.002329107f,	0.001851353f,	0.001342483f,	0.000888072f,	0.000535929f,	0.000295044f,	0.000148179f,	6.78899E-05f,	2.83755E-05f,	0.0f,
	0.001540738f,	0.001471597f,	0.001282238f,	0.001019221f,	0.000739074f,	0.000488908f,	0.000295044f,	0.00016243f,	8.15765E-05f,	3.73753E-05f,	1.56215E-05f,	0.0f,
	0.000773799f,	0.000739074f,	0.000643973f,	0.000511879f,	0.000371182f,	0.000245542f,	0.000148179f,	8.15765E-05f,	4.09698E-05f,	1.87708E-05f,	7.84553E-06f,	0.0f,
	0.000354525f,	0.000338616f,	0.000295044f,	0.000234524f,	0.000170062f,	0.000112498f,	6.78899E-05f,	3.73753E-05f,	1.87708E-05f,	8.60008E-06f,	3.59452E-06f,	0.0f,
	0.000148179f,	0.000141529f,	0.000123318f,	9.80224E-05f,	7.10796E-05f,	4.70202E-05f,	2.83755E-05f,	1.56215E-05f,	7.84553E-06f,	3.59452E-06f,	1.50238E-06f,	0.0f,
	0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f,			0.0f
};

template<int w, int h>
__global__ void computeHessianDeterminants(float *d_hd, cudaTextureObject_t t_rii);
template<int w, int h>
__global__ void supressInterestPoints(int *d_ip, int *d_ipc, float *d_hd);
template<int w, int h>
__global__ void refineInterestPoints(float *d_rip, int *d_ripc, float *d_H, int *d_ip, int d_ipc);
__global__ void describeInterestPoints(float *d_ipd, float *d_hd, float *d_rip, cudaTextureObject_t t_rii);
__global__ void normalizeInterestPointDescriptors(float *d_ipd, float *d_hd);
__global__ void matchInterestPoints(int *d_ipm, int *d_ipmc, float *d_ipd1, float *d_ipd2, int d_ripc1, int d_ripc2);

__device__ float boxFilter(int d_r, int d_c, int d_w, int d_h, cudaTextureObject_t t_rii);
__device__ float horizontalFirstOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii);
__device__ float verticalFirstOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii);
__device__ float horizontalSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii);
__device__ float verticalSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii);
__device__ float mixedSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii);

void initializeFeatureMatching() {
	cudaMalloc(&d_lhd, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * COMPUTATION_SCALES * sizeof(float));
	cudaMalloc(&d_rhd, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * COMPUTATION_SCALES * sizeof(float));
	cudaMalloc(&d_lip, MAXIMUM_INTEREST_POINTS * sizeof(int) * 3);
	cudaMalloc(&d_lipc, sizeof(int));
	cudaMalloc(&d_rip, MAXIMUM_INTEREST_POINTS * sizeof(int) * 3);
	cudaMalloc(&d_ripc, sizeof(int));
	cudaMalloc(&d_lrip, MAXIMUM_INTEREST_POINTS * sizeof(int) * 3);
	cudaMalloc(&d_lripc, sizeof(int));
	cudaMalloc(&d_rrip, MAXIMUM_INTEREST_POINTS * sizeof(int) * 3);
	cudaMalloc(&d_rripc, sizeof(int));
	cudaMalloc(&d_lipd, MAXIMUM_INTEREST_POINTS * DESCRIPTOR_DIMENSIONS * sizeof(int));
	cudaMalloc(&d_ripd, MAXIMUM_INTEREST_POINTS * DESCRIPTOR_DIMENSIONS * sizeof(int));
	cudaMalloc(&d_ipm, MAXIMUM_INTEREST_POINTS * sizeof(int) * 2);
	cudaMalloc(&d_ipmc, sizeof(int));
}

void computeInterestPoints() {
	computeHessianDeterminants<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<COMPUTATION_BLOCKS, COMPUTATION_THREADS>>>(d_lhd, t_lriii);
	cudaMemset(&d_lipc, 0, sizeof(int));
	supressInterestPoints<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<SUPPRESSION_BLOCKS, SUPPRESSION_THREADS>>>(d_lip, d_lipc, d_lhd);
	cudaMemcpy(&lipc, d_lipc, sizeof(int), cudaMemcpyDeviceToHost);
	lipc = std::min(lipc, MAXIMUM_INTEREST_POINTS);
	cudaMemset(&d_lripc, 0, sizeof(int));
	refineInterestPoints<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<(lipc + REFINEMENT_THREADS - 1) / REFINEMENT_THREADS, REFINEMENT_THREADS>>>(d_lrip, d_lripc, d_lhd, d_lip, lipc);
	cudaMemcpy(&lripc, d_lripc, sizeof(int), cudaMemcpyDeviceToHost);
	describeInterestPoints<<<lripc, dim3(DESCRIPTOR_BLOCK_SIZE, DESCRIPTOR_BLOCK_SIZE, DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE)>>>(d_lipd, d_lhd, d_lrip, t_lriii);
	normalizeInterestPointDescriptors<<<lripc, DESCRIPTOR_DIMENSIONS>>>(d_lipd, d_lhd);
	computeHessianDeterminants<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<COMPUTATION_BLOCKS, COMPUTATION_THREADS>>>(d_rhd, t_rriii);
	cudaMemset(d_ripc, 0, sizeof(int));
	supressInterestPoints<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<SUPPRESSION_BLOCKS, SUPPRESSION_THREADS>>>(d_rip, d_ripc, d_rhd);
	cudaMemcpy(&ripc, d_ripc, sizeof(int), cudaMemcpyDeviceToHost);
	ripc = std::min(ripc, MAXIMUM_INTEREST_POINTS);
	cudaMemset(d_rripc, 0, sizeof(int));
	refineInterestPoints<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<(ripc + REFINEMENT_THREADS - 1) / REFINEMENT_THREADS, REFINEMENT_THREADS>>>(d_rrip, d_rripc, d_rhd, d_rip, ripc);
	cudaMemcpy(&rripc, d_rripc, sizeof(int), cudaMemcpyDeviceToHost);
	describeInterestPoints<<<rripc, dim3(DESCRIPTOR_BLOCK_SIZE, DESCRIPTOR_BLOCK_SIZE, DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE)>>>(d_ripd, d_rhd, d_rrip, t_rriii);
	normalizeInterestPointDescriptors<<<rripc, DESCRIPTOR_DIMENSIONS>>>(d_ripd, d_rhd);
	cudaMemset(d_ipmc, 0, sizeof(int));
}

void matchInterestPoints() {
	cudaMemset(d_ipmc, 0, sizeof(int));
	matchInterestPoints<<<(rripc + MATCHING_THREAD_INTEREST_POINTS - 1) / MATCHING_THREAD_INTEREST_POINTS, dim3(DESCRIPTOR_DIMENSIONS / 2, MATCHING_THREAD_INTEREST_POINTS)>>>(d_ipm, d_ipmc, d_lipd, d_ripd, lipc, ripc);
	cudaMemcpy(&ipmc, d_ipmc, sizeof(int), cudaMemcpyDeviceToHost);
}

template<int w, int h>
__global__ void computeHessianDeterminants(float *d_hd, cudaTextureObject_t t_rii) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / w;
	l_c -= l_r * w;
	if (l_c > COMPUTATION_PADDING - 1 && l_c < w - COMPUTATION_PADDING && l_r > COMPUTATION_PADDING - 1 && l_r < h - COMPUTATION_PADDING) {
		#pragma unroll
		for (int l_s = 0; l_s < COMPUTATION_SCALES; l_s++) {
			int l_ss = d_css[l_s];
			if ((l_c - COMPUTATION_PADDING) % l_ss != 0 || (l_r - COMPUTATION_PADDING) % l_ss != 0)
				return;
			float l_fhdt = 1.0f / d_l[l_s];
			l_fhdt *= l_fhdt;
			float l_shdt = l_fhdt * mixedSecondOrderFilter(l_r, l_c, d_l[l_s], t_rii) * 0.912f;
			l_fhdt *= l_fhdt;
			l_fhdt *= horizontalSecondOrderFilter(l_r, l_c, d_l[l_s], t_rii) * verticalSecondOrderFilter(l_r, l_c, d_l[l_s], t_rii);
			l_shdt *= l_shdt;
			d_hd[l_c + l_r * w + l_s * w * h] = l_fhdt - l_shdt;
		}
	}
}

template<int w, int h>
__global__ void supressInterestPoints(int *d_ip, int *d_ipc, float *d_hd) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / w;
	l_c -= l_r * w;
	if (l_c > COMPUTATION_PADDING - 1 && l_c < w - COMPUTATION_PADDING && l_r > COMPUTATION_PADDING - 1 && l_r < h - COMPUTATION_PADDING) {
		for (int l_s = 1; l_s < COMPUTATION_SCALES - 1; l_s++) {
			int l_ss = d_sss[l_s];
			if ((l_c - COMPUTATION_PADDING) % l_ss != 0 || (l_r - COMPUTATION_PADDING) % l_ss != 0)
				return;
			float l_hd = d_hd[l_c + l_r * w + l_s * w * h];
			if (l_hd < SUPRESSION_THRESHOLD)
				continue;
			float l_mhd = FLT_MIN;
			for (int l_so = 0; l_so < 3; l_so++)
				for (int l_ro = -l_ss; l_ro < l_ss + 1 && l_hd >= l_mhd; l_ro += l_ss)
					#pragma unroll
					for (int l_co = -l_ss; l_co < l_ss + 1 && l_hd >= l_mhd; l_co += l_ss)
						l_mhd = fmaxf(l_mhd, d_hd[l_c + l_co + (l_r + l_ro) * w + (l_s + d_sso[l_s * 3 + l_so]) * w * h]);
			if (l_hd == l_mhd) {
				int l_ipc = atomicAdd(d_ipc, 1);
				if (l_ipc < MAXIMUM_INTEREST_POINTS) {
					d_ip[l_ipc] = l_r;
					d_ip[l_ipc + MAXIMUM_INTEREST_POINTS] = l_c;
					d_ip[l_ipc + 2 * MAXIMUM_INTEREST_POINTS] = l_s;
				}
			}
		}
	}
}

#define hd(rd, cd, sd) d_hd[l_c + cd + (l_r + rd) * w + (l_s + d_sso[l_s * 3 + sd + 1]) * w * h]

template<int w, int h>
__global__ void refineInterestPoints(float *d_rip, int *d_ripc, float *d_hd, int *d_ip, int d_ipc) {
	int l_ip = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_ip < d_ipc) {
		int l_r = d_ip[l_ip];
		int l_c = d_ip[l_ip + MAXIMUM_INTEREST_POINTS];
		int l_s = d_ip[l_ip + 2 * MAXIMUM_INTEREST_POINTS];
		int l_ss = d_sss[l_s];
		float l_am[12];
		l_am[3] = (hd(0, l_ss, 0) - hd(0, -l_ss, 0)) / (2.0f * l_ss);
		l_am[7] = (hd(l_ss, 0, 0) - hd(-l_ss, 0, 0)) / (2.0f * l_ss);
		l_am[11] = (hd(0, 0, 1) - hd(0, 0, -1)) / (4.0f * l_ss);
		l_am[0] = (hd(0, l_ss, 0) - 2.0f * hd(0, 0, 0) + hd(0, -l_ss, 0)) / (l_ss * l_ss);
		l_am[5] = (hd(l_ss, 0, 0) - 2.0f * hd(0, 0, 0) + hd(-l_ss, 0, 0)) / (l_ss * l_ss);
		l_am[10] = (hd(0, 0, 1) - 2.0f * hd(0, 0, 0) + hd(0, 0, -1)) / (4.0f * l_ss * l_ss);
		l_am[1] = (hd(l_ss, l_ss, 0) + hd(-l_ss, -l_ss, 0) - hd(l_ss, -l_ss, 0) - hd(-l_ss, l_ss, 0)) / (4.0f * l_ss * l_ss);
		l_am[2] = (hd(0, l_ss, 1) + hd(0, -l_ss, -1) - hd(0, -l_ss, 1) - hd(0, l_ss, -1)) / (8.0f * l_ss * l_ss);
		l_am[6] = (hd(l_ss, 0, 1) + hd(-l_ss, 0, -1) - hd(-l_ss, 0, 1) - hd(l_ss, 0, -1)) / (8.0f * l_ss * l_ss);
		l_am[4] = l_am[1];
		l_am[8] = l_am[2];
		l_am[9] = l_am[6];
		int l_yr, l_zr;
		for (int l_mr = 0; l_mr < 3; l_mr++) {
			float l_d = l_am[l_mr * 4];
			if (fabs(l_d) > REFINEMENT_EPSILON)
				for (int l_co = 0; l_co < 4; l_co++)
					l_am[l_co + l_mr * 4] /= l_d;
		}
		if (fabsf(l_am[0]) > REFINEMENT_EPSILON) {
			if (fabsf(l_am[4]) > REFINEMENT_EPSILON)
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + 4] -= l_am[l_mc];
			if (fabsf(l_am[8]) > REFINEMENT_EPSILON)
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + 8] -= l_am[l_mc];
			l_yr = 1;
			l_zr = 2;
		} else if (fabsf(l_am[4]) > REFINEMENT_EPSILON) {
			if (fabsf(l_am[8]) > REFINEMENT_EPSILON)
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + 8] -= l_am[l_mc + 4];
			l_yr = 0;
			l_zr = 2;
		} else if (fabsf(l_am[8]) > REFINEMENT_EPSILON) {
			l_yr = 0;
			l_zr = 1;
		} else
			return;
		if (fabsf(l_am[1 + l_zr * 4]) > REFINEMENT_EPSILON) {
			if (fabs(l_am[1 + l_yr * 4]) > REFINEMENT_EPSILON) {
				float l_d = l_am[1 + l_yr * 4];
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + l_yr * 4] /= l_d;
				l_d = l_am[1 + l_zr * 4];
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + l_zr * 4] /= l_d;
				for (int l_mc = 0; l_mc < 4; l_mc++)
					l_am[l_mc + l_zr * 4] -= l_am[l_mc + l_yr * 4];
			} else {
				int l_tr = l_yr;
				l_yr = l_zr;
				l_zr = l_tr;
			}
		} else if (!(fabs(l_am[1 + l_yr * 4]) > REFINEMENT_EPSILON))
			return;
		float l_sd = l_am[3 + l_zr * 4] / l_am[2 + l_zr * 4];
		float l_cd = (l_am[3 + l_yr * 4] - l_am[2 + l_yr * 4] * l_sd) / l_am[1 + l_yr * 4];
		float l_rd = (l_am[3 + (3 - l_yr - l_zr) * 4] - l_am[1 + (3 - l_yr - l_zr) * 4] * l_cd - l_am[2 + (3 - l_yr - l_zr) * 4] * l_sd) / l_am[(3 - l_yr - l_zr) * 4];
		if (fmaxf(fmaxf(fabsf(l_rd), fabsf(l_cd)), 0.5f * fabsf(l_sd)) < l_ss) {
			int l_ripc = atomicAdd(d_ripc, 1);
			d_rip[l_ripc] = l_r - l_rd;
			d_rip[l_ripc + MAXIMUM_INTEREST_POINTS] = l_c - l_cd;
			d_rip[l_ripc + 2 * MAXIMUM_INTEREST_POINTS] = d_l[l_s] - l_sd;
		}
	}
}

__global__ void describeInterestPoints(float *d_ipd, float *d_hd, float *d_rip, cudaTextureObject_t t_rii) {
	int l_br = threadIdx.z / DESCRIPTOR_GRID_SIZE;
	int l_bc = threadIdx.z - l_br * DESCRIPTOR_GRID_SIZE;
	int l_ro = threadIdx.y + l_br * DESCRIPTOR_BLOCK_SIZE - DESCRIPTOR_BLOCK_SIZE * DESCRIPTOR_GRID_SIZE / 2;
	int l_co = threadIdx.x + l_bc * DESCRIPTOR_BLOCK_SIZE - DESCRIPTOR_BLOCK_SIZE * DESCRIPTOR_GRID_SIZE / 2;
	float l_s = d_rip[blockIdx.x + 2 * MAXIMUM_INTEREST_POINTS] * DESCRIPTOR_SIGMA;
	int l_sr = llrintf(d_rip[blockIdx.x] + l_s * l_ro);
	int l_sc = llrintf(d_rip[blockIdx.x + MAXIMUM_INTEREST_POINTS] + l_s * l_co);
	float l_w = d_gk[abs(l_ro)][abs(l_co)];
	volatile __shared__ float s_hfof[DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE][DESCRIPTOR_BLOCK_SIZE][DESCRIPTOR_BLOCK_SIZE];
	volatile __shared__ float s_vfof[DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE][DESCRIPTOR_BLOCK_SIZE][DESCRIPTOR_BLOCK_SIZE];
	s_hfof[threadIdx.z][threadIdx.x][threadIdx.y] = horizontalFirstOrderFilter(l_sr, l_sc, llrintf(l_s), t_rii) * l_w;
	s_vfof[threadIdx.z][threadIdx.x][threadIdx.y] = verticalFirstOrderFilter(l_sr, l_sc, llrintf(l_s), t_rii) * l_w;
	__syncthreads();
	volatile __shared__ float s_s[DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE][DESCRIPTOR_BLOCK_DIMENSIONS][DESCRIPTOR_BLOCK_SIZE];
	if (threadIdx.y == 0) {
		float l_shfof = 0.0f;
		float l_svfof = 0.0f;
		float l_ashfof = 0.0f;
		float l_asvfof = 0.0f;
		for (int l_r = 0; l_r < DESCRIPTOR_BLOCK_SIZE; l_r++) {
			l_shfof += s_hfof[threadIdx.z][threadIdx.x][l_r];
			l_svfof += s_vfof[threadIdx.z][threadIdx.x][l_r];
			l_ashfof += fabsf(s_hfof[threadIdx.z][threadIdx.x][l_r]);
			l_asvfof += fabsf(s_vfof[threadIdx.z][threadIdx.x][l_r]);	
		}
		s_s[threadIdx.z][0][threadIdx.x] = l_shfof;
		s_s[threadIdx.z][1][threadIdx.x] = l_svfof;
		s_s[threadIdx.z][2][threadIdx.x] = l_ashfof;
		s_s[threadIdx.z][3][threadIdx.x] = l_asvfof;
	}
	__syncthreads();
	volatile __shared__ float s_d[DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE][DESCRIPTOR_BLOCK_DIMENSIONS];
	if (threadIdx.x < DESCRIPTOR_BLOCK_DIMENSIONS && threadIdx.y == 0) {
		float l_c = s_s[threadIdx.z][threadIdx.x][0] + s_s[threadIdx.z][threadIdx.x][1] + s_s[threadIdx.z][threadIdx.x][2] + s_s[threadIdx.z][threadIdx.x][3];
		s_d[threadIdx.z][threadIdx.x] = l_c;
		d_ipd[threadIdx.x + threadIdx.z * DESCRIPTOR_BLOCK_DIMENSIONS + blockIdx.x * DESCRIPTOR_DIMENSIONS] = l_c;
	}
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
		d_hd[threadIdx.z + blockIdx.x * DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE] = s_d[threadIdx.z][0] * s_d[threadIdx.z][0] + s_d[threadIdx.z][1] * s_d[threadIdx.z][1] + s_d[threadIdx.z][2] * s_d[threadIdx.z][2] + s_d[threadIdx.z][3] * s_d[threadIdx.z][3];
}

__global__ void normalizeInterestPointDescriptors(float *d_ipd, float *d_hd) {
	int l_ei = threadIdx.x * DESCRIPTOR_GRID_SIZE + blockIdx.x * DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE;
	volatile __shared__ float s_ss[DESCRIPTOR_GRID_SIZE];
	if (threadIdx.x < DESCRIPTOR_GRID_SIZE)
		s_ss[threadIdx.x] = d_hd[l_ei] + d_hd[l_ei + 1] + d_hd[l_ei + 2] + d_hd[l_ei + 3];
	__syncthreads();
	d_ipd[threadIdx.x + blockIdx.x * DESCRIPTOR_DIMENSIONS] *= rsqrtf(s_ss[0] + s_ss[1] + s_ss[2] + s_ss[3]);
}

__global__ void matchInterestPoints(int *d_ipm, int *d_ipmc, float *d_ipd1, float *d_ipd2, int d_ripc1, int d_ripc2) {
	int l_ip1 = threadIdx.y + blockIdx.x * MATCHING_THREAD_INTEREST_POINTS;
	float l_e1;
	float l_e2;
	if (l_ip1 < d_ripc1) {
		l_e1 = d_ipd1[threadIdx.x + l_ip1 * DESCRIPTOR_DIMENSIONS];
		l_e2 = d_ipd1[threadIdx.x + DESCRIPTOR_DIMENSIONS / 2 + l_ip1 * DESCRIPTOR_DIMENSIONS];
	}
	volatile __shared__ float s_d1[DESCRIPTOR_DIMENSIONS * MATCHING_THREAD_INTEREST_POINTS];
	volatile __shared__ float s_d2[DESCRIPTOR_DIMENSIONS];
	float l_bd = FLT_MAX;
	float l_sbd = FLT_MAX;
	int l_bip = -1;
	int l_ei1 = threadIdx.y * DESCRIPTOR_DIMENSIONS;
	int l_ei2 = threadIdx.x + threadIdx.y * DESCRIPTOR_DIMENSIONS / 2;
	for (int l_ip2 = 0; l_ip2 < d_ripc2; l_ip2++) {

		if (l_ei2 < DESCRIPTOR_DIMENSIONS)
			s_d2[l_ei2] = d_ipd2[l_ei2 + l_ip2 * DESCRIPTOR_DIMENSIONS];
		__syncthreads();
		float l_d = l_e1 - s_d2[threadIdx.x];
		s_d1[threadIdx.x + l_ei1] = l_d * l_d;
		l_d = l_e2 - s_d2[threadIdx.x + DESCRIPTOR_DIMENSIONS / 2];
		s_d1[threadIdx.x + DESCRIPTOR_DIMENSIONS / 2 + l_ei1] = l_d * l_d;
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 32];
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 16];
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 8];
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 4];
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 2];
		s_d1[threadIdx.x + l_ei1] += s_d1[threadIdx.x + l_ei1 + 1];
		__syncthreads();
		l_d = sqrtf(s_d1[l_ei1]);
		if (l_d < l_bd) {
			l_sbd = l_bd;
			l_bd = l_d;
			l_bip = l_ip2;
		} else if (l_d < l_sbd)
			l_sbd = l_d;
	}
	if (threadIdx.x == 0 && l_ip1 < d_ripc1 && l_bd < MATCHING_DISTANCE_THRESHOLD && l_bd / l_sbd < MATCHING_DISTANCE_RATIO_THRESHOLD) {
		int l_ipmc = atomicAdd(d_ipmc, 1);
		d_ipm[l_ipmc] = l_ip1;
		d_ipm[l_ipmc + MAXIMUM_INTEREST_POINTS] = l_bip;
	}
}

__device__ float boxFilter(int d_r, int d_c, int d_w, int d_h, cudaTextureObject_t t_rii) {
	return (float) tex2D<int>(t_rii, d_c + 1, d_r + 1) - tex2D<int>(t_rii, d_c + 1 - d_w, d_r + 1) - tex2D<int>(t_rii, d_c + 1, d_r + 1 - d_h) + tex2D<int>(t_rii, d_c + 1 - d_w, d_r + 1 - d_h);
}

__device__ float horizontalFirstOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii) {
	return boxFilter(d_r + d_l, d_c - 1, d_l, 2 * d_l + 1, t_rii) - boxFilter(d_r + d_l, d_c + d_l, d_l, 2 * d_l + 1, t_rii);
}

__device__ float verticalFirstOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii) {
	return boxFilter(d_r - 1, d_c + d_l, 2 * d_l + 1, d_l, t_rii) - boxFilter(d_r + d_l, d_c + d_l, 2 * d_l + 1, d_l, t_rii);
}

__device__ float horizontalSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii) {
	return boxFilter(d_r + d_l, d_c + d_l + d_l / 2, 3 * d_l, 2 * d_l + 1, t_rii) - 3.0f * boxFilter(d_r + d_l, d_c + d_l / 2, d_l, 2 * d_l + 1, t_rii);
}

__device__ float verticalSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii) {
	return boxFilter(d_r + d_l + d_l / 2, d_c + d_l, 2 * d_l + 1, 3 * d_l, t_rii) - 3.0f * boxFilter(d_r + d_l / 2, d_c + d_l, 2 * d_l + 1, d_l, t_rii);
}

__device__ float mixedSecondOrderFilter(int d_r, int d_c, int d_l, cudaTextureObject_t t_rii) {
	return boxFilter(d_r + d_l, d_c - 1, d_l, d_l, t_rii) + boxFilter(d_r - 1, d_c + d_l, d_l, d_l, t_rii) - boxFilter(d_r + d_l, d_c + d_l, d_l, d_l, t_rii) - boxFilter(d_r - 1, d_c - 1, d_l, d_l, t_rii);
}