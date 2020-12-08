#include <cuda_runtime.h>

#include "cam.hpp"

#define RECTIFICATION_THREADS 128
#define RECTIFICATION_THREAD_PIXELS 16
#define RECTIFICATION_BLOCKS (((RECTIFIED_IMAGE_HEIGHT + RECTIFICATION_THREAD_PIXELS - 1) / (RECTIFICATION_THREAD_PIXELS) * RECTIFIED_IMAGE_WIDTH + RECTIFICATION_THREADS - 1) / RECTIFICATION_THREADS)

#define TRANSPOSING_THREADS 128
#define TRANSPOSING_THREAD_PIXELS 16
#define TRANSPOSING_BLOCKS (((RECTIFIED_IMAGE_WIDTH + TRANSPOSING_THREAD_PIXELS - 1) / (TRANSPOSING_THREAD_PIXELS) * RECTIFIED_IMAGE_HEIGHT + TRANSPOSING_THREADS - 1) / TRANSPOSING_THREADS)

unsigned char *d_di;
size_t d_dip;

unsigned char *d_lri;
unsigned char *d_rri;

unsigned char *d_ltri;
unsigned char *d_rtri;

texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t_di;
size_t t_dio;

template<int w, int h>
__global__ void rectifyLeftImage(unsigned char *d_ri, unsigned char *d_di, int t_dio);
template<int w, int h>
__global__ void rectifyRightImage(unsigned char *d_ri, unsigned char *d_di, int t_dio);
template<int w, int h>
__global__ void transposeImage(unsigned char *d_tri, unsigned char *d_ri);

void initializePreprocessing() {
	cudaMallocPitch(&d_di, &d_dip, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT);
	d_dip /= sizeof(unsigned char);
	t_di.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc c = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(&t_dio, &t_di, d_di, &c, DISTORTED_IMAGE_WIDTH, DISTORTED_IMAGE_HEIGHT, d_dip);
	t_dio /= sizeof(unsigned char);
	cudaMalloc(&d_lri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMalloc(&d_rri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMalloc(&d_ltri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMalloc(&d_rtri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
}

void rectifyImages(unsigned char *l, unsigned char *r, unsigned char *o) {
	cudaMemcpy2D(d_di, d_dip, l, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	rectifyLeftImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECTIFICATION_BLOCKS, RECTIFICATION_THREADS>>>(d_lri, d_di, t_dio);
	cudaMemcpy2D(d_di, d_dip, r, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	rectifyRightImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECTIFICATION_BLOCKS, RECTIFICATION_THREADS>>>(d_rri, d_di, t_dio);
}

void transposeRectifiedImages() {
	transposeImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<TRANSPOSING_BLOCKS, TRANSPOSING_THREADS>>>(d_ltri, d_lri);
	transposeImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<TRANSPOSING_BLOCKS, TRANSPOSING_THREADS>>>(d_rtri, d_rri);
}

template<int w, int h>
__global__ void rectifyLeftImage(unsigned char *d_ri, unsigned char *d_di, int t_dio) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / w;
	l_c -= l_r * w;
	l_r *= RECTIFICATION_THREAD_PIXELS;
	for (int l_ro = 0; l_ro < RECTIFICATION_THREAD_PIXELS && l_r + l_ro < h; l_ro++) {
		float l_ud = (l_c + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_20 + (l_r + l_ro + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_21 + LEFT_HOMOGRAPHY_22;
		float l_ux = ((l_c + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_00 + (l_r + l_ro + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_01 + LEFT_HOMOGRAPHY_02) / l_ud;
		float l_uy = ((l_c + LEFT_RECTIFIED_COLUMN) * LEFT_HOMOGRAPHY_10 + (l_r + l_ro + TOP_RECTIFIED_ROW) * LEFT_HOMOGRAPHY_11 + LEFT_HOMOGRAPHY_12) / l_ud;
		float l_sr = l_ux * l_ux + l_uy * l_uy;
		float l_df = l_sr * l_sr * SECOND_LEFT_DISTORTION_COEFFICIENT + l_sr * FIRST_LEFT_DISTORTION_COEFFICIENT + 1.0;
		float l_dc = l_ux * l_df * LEFT_HORIZONTAL_FOCAL_LENGTH + LEFT_HORIZONTAL_PRINCIPAL_POINT;
		float l_dr = l_uy * l_df * LEFT_VERTICAL_FOCAL_LENGTH + LEFT_VERTICAL_PRINCIPAL_POINT;
		d_ri[l_c + (l_r + l_ro) * w] = (unsigned char) (tex2D(t_di, l_dc + t_dio, l_dr) * UCHAR_MAX);
	}
}

template<int w, int h>
__global__ void rectifyRightImage(unsigned char *d_ri, unsigned char *d_di, int t_dio) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	int l_r = l_c / w;
	l_c -= l_r * w;
	l_r *= RECTIFICATION_THREAD_PIXELS;
	for (int l_ro = 0; l_ro < RECTIFICATION_THREAD_PIXELS && l_r + l_ro < h; l_ro++) {
		float l_ud = (l_c + LEFT_RECTIFIED_COLUMN) * RIGHT_HOMOGRAPHY_20 + (l_r + l_ro + TOP_RECTIFIED_ROW) * RIGHT_HOMOGRAPHY_21 + RIGHT_HOMOGRAPHY_22;
		float l_ux = ((l_c + LEFT_RECTIFIED_COLUMN) * RIGHT_HOMOGRAPHY_00 + (l_r + l_ro + TOP_RECTIFIED_ROW) * RIGHT_HOMOGRAPHY_01 + RIGHT_HOMOGRAPHY_02) / l_ud;
		float l_uy = ((l_c + LEFT_RECTIFIED_COLUMN) * RIGHT_HOMOGRAPHY_10 + (l_r + l_ro + TOP_RECTIFIED_ROW) * RIGHT_HOMOGRAPHY_11 + RIGHT_HOMOGRAPHY_12) / l_ud;
		float l_sr = l_ux * l_ux + l_uy * l_uy;
		float l_df = l_sr * l_sr * SECOND_RIGHT_DISTORTION_COEFFICIENT + l_sr * FIRST_RIGHT_DISTORTION_COEFFICIENT + 1.0;
		float l_dc = l_ux * l_df * RIGHT_HORIZONTAL_FOCAL_LENGTH + RIGHT_HORIZONTAL_PRINCIPAL_POINT;
		float l_dr = l_uy * l_df * RIGHT_VERTICAL_FOCAL_LENGTH + RIGHT_VERTICAL_PRINCIPAL_POINT;
		d_ri[l_c + (l_r + l_ro) * w] = (unsigned char) (tex2D(t_di, l_dc + t_dio, l_dr) * UCHAR_MAX);
	}
}

template<int w, int h>
__global__ void transposeImage(unsigned char *d_tri, unsigned char *d_ri) {
	int l_r = threadIdx.x + blockIdx.x * blockDim.x;
	int l_c = l_r / h;
	l_r -= l_c * h;
	l_c *= TRANSPOSING_THREAD_PIXELS;
	#pragma unroll
	for (int l_ro = 0; l_ro < TRANSPOSING_THREAD_PIXELS && l_c + l_ro < w; l_ro++)
		d_tri[l_r + (l_c + l_ro) * h] = d_ri[l_c + l_ro + l_r * w];
}