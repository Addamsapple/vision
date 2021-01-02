#include <cuda_runtime.h>

#include "cam.hpp"

#define RECTIFICATION_THREADS 128
#define RECTIFICATION_THREAD_PIXELS 16
#define RECTIFICATION_BLOCKS (((RECTIFIED_IMAGE_HEIGHT + RECTIFICATION_THREAD_PIXELS - 1) / (RECTIFICATION_THREAD_PIXELS) * RECTIFIED_IMAGE_WIDTH + RECTIFICATION_THREADS - 1) / RECTIFICATION_THREADS)

#define TRANSPOSING_THREADS 128
#define TRANSPOSING_THREAD_PIXELS 16
#define TRANSPOSING_BLOCKS (((RECTIFIED_IMAGE_WIDTH + TRANSPOSING_THREAD_PIXELS - 1) / (TRANSPOSING_THREAD_PIXELS) * RECTIFIED_IMAGE_HEIGHT + TRANSPOSING_THREADS - 1) / TRANSPOSING_THREADS)

#define VERTICAL_INTEGRATION_THREADS 128
#define VERTICAL_INTEGRATION_BLOCKS ((RECTIFIED_IMAGE_WIDTH + 1 + VERTICAL_INTEGRATION_THREADS - 1) / VERTICAL_INTEGRATION_THREADS)
#define HORIZONTAL_INTEGRATION_THREADS 416
#define HORIZONTAL_INTEGRATION_BLOCKS RECTIFIED_IMAGE_HEIGHT

unsigned char *d_di;
size_t d_dip;
texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> t_di;
size_t t_dio;

unsigned char *d_lri;
unsigned char *d_rri;

int *d_lrii;
size_t d_lriip;
cudaTextureObject_t t_lriii;

int *d_rrii;
size_t d_rriip;
cudaTextureObject_t t_rriii;

unsigned char *d_ltri;
unsigned char *d_rtri;

template<int w, int h>
__global__ void rectifyLeftImage(unsigned char *d_ri, unsigned char *d_di, int t_dio);
template<int w, int h>
__global__ void rectifyRightImage(unsigned char *d_ri, unsigned char *d_di, int t_dio);
template<int w, int h>
__global__ void integrateImageVertically(int *d_rii, int d_riip, unsigned char *d_ri);
template<int w>
__global__ void integrateImageHorizontally(int *d_rii, int d_riip);
template<int w, int h>
__global__ void transposeImage(unsigned char *d_tri, unsigned char *d_ri);

#include <iostream>

void initializePreprocessing() {
	cudaMallocPitch(&d_di, &d_dip, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT);
	t_di.addressMode[0] = cudaAddressModeBorder;
	t_di.addressMode[1] = cudaAddressModeBorder;
	t_di.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc textureChannelDescription = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(&t_dio, &t_di, d_di, &textureChannelDescription, DISTORTED_IMAGE_WIDTH, DISTORTED_IMAGE_HEIGHT, d_dip);
	d_dip /= sizeof(unsigned char);
	t_dio /= sizeof(unsigned char);
	cudaMalloc(&d_lri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMalloc(&d_rri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMallocPitch(&d_lrii, &d_lriip, (RECTIFIED_IMAGE_WIDTH + 1) * sizeof(int), RECTIFIED_IMAGE_HEIGHT + 1);
	textureChannelDescription = cudaCreateChannelDesc<int>();
	cudaResourceDesc textureResourceDescription;
	memset(&textureResourceDescription, 0, sizeof(textureResourceDescription));
	textureResourceDescription.resType = cudaResourceTypePitch2D;
	textureResourceDescription.res.pitch2D.devPtr = d_lrii;
	textureResourceDescription.res.pitch2D.desc = textureChannelDescription;
	textureResourceDescription.res.pitch2D.width = RECTIFIED_IMAGE_WIDTH + 1;
	textureResourceDescription.res.pitch2D.height = RECTIFIED_IMAGE_HEIGHT + 1;
	textureResourceDescription.res.pitch2D.pitchInBytes = d_lriip;
	d_lriip /= sizeof(int);
	cudaTextureDesc textureDescription;
	memset(&textureDescription, 0, sizeof(textureDescription));
	textureDescription.addressMode[0] = cudaAddressModeClamp;
	textureDescription.addressMode[1] = cudaAddressModeClamp;
	textureDescription.filterMode = cudaFilterModePoint;
	textureDescription.readMode = cudaReadModeElementType;
	textureDescription.normalizedCoords = false;
	memset(&t_lriii, 0, sizeof(t_lriii));
	cudaCreateTextureObject(&t_lriii, &textureResourceDescription, &textureDescription, NULL);
	cudaMallocPitch(&d_rrii, &d_rriip, (RECTIFIED_IMAGE_WIDTH + 1) * sizeof(int), RECTIFIED_IMAGE_HEIGHT + 1);
	textureResourceDescription.res.pitch2D.devPtr = d_rrii;
	textureResourceDescription.res.pitch2D.pitchInBytes = d_rriip;
	d_rriip /= sizeof(int);
	memset(&t_rriii, 0, sizeof(t_rriii));
	cudaCreateTextureObject(&t_rriii, &textureResourceDescription, &textureDescription, NULL);
	cudaMalloc(&d_ltri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
	cudaMalloc(&d_rtri, RECTIFIED_IMAGE_WIDTH * RECTIFIED_IMAGE_HEIGHT * sizeof(unsigned char));
}

void rectifyImages(unsigned char *l, unsigned char *r, unsigned char *o) {
	cudaMemcpy2D(d_di, d_dip, l, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	rectifyLeftImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECTIFICATION_BLOCKS, RECTIFICATION_THREADS>>>(d_lri, d_di, t_dio);
	cudaMemcpy2D(d_di, d_dip, r, DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_WIDTH * sizeof(unsigned char), DISTORTED_IMAGE_HEIGHT, cudaMemcpyHostToDevice);
	rectifyRightImage<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<RECTIFICATION_BLOCKS, RECTIFICATION_THREADS>>>(d_rri, d_di, t_dio);
}

void integrateImages() {
	integrateImageVertically<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<VERTICAL_INTEGRATION_BLOCKS, VERTICAL_INTEGRATION_THREADS>>>(d_lrii, d_lriip, d_lri);
	integrateImageHorizontally<RECTIFIED_IMAGE_WIDTH><<<HORIZONTAL_INTEGRATION_BLOCKS, HORIZONTAL_INTEGRATION_THREADS>>>(d_lrii, d_lriip);  
	integrateImageVertically<RECTIFIED_IMAGE_WIDTH, RECTIFIED_IMAGE_HEIGHT><<<VERTICAL_INTEGRATION_BLOCKS, VERTICAL_INTEGRATION_THREADS>>>(d_rrii, d_rriip, d_rri);
	integrateImageHorizontally<RECTIFIED_IMAGE_WIDTH><<<HORIZONTAL_INTEGRATION_BLOCKS, HORIZONTAL_INTEGRATION_THREADS>>>(d_rrii, d_rriip);	
}

void transposeImages() {
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
		//offset texture coordinates by 0.5?
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
__global__ void integrateImageVertically(int *d_rii, int d_riip, unsigned char *d_ri) {
	int l_c = threadIdx.x + blockIdx.x * blockDim.x;
	if (l_c < w + 1) {
		d_rii[l_c] = 0;
		int l_s = 0;
		if (l_c > 0)
			for (int l_r = 1; l_r < h + 1; l_r++) {
				l_s += d_ri[l_c - 1 + (l_r - 1) * w];
				d_rii[l_c + l_r * d_riip] = l_s;
			}
		else
			for (int l_r = 1; l_r < h + 1; l_r++)
				d_rii[l_r * d_riip] = 0;
	}
}

template<int w>
__global__ void integrateImageHorizontally(int *d_rii, int d_riip) {
	int l_r = blockIdx.x + 1;
	volatile __shared__ int s_b[2][w + 1];
	for (int l_c = threadIdx.x; l_c < w + 1; l_c += blockDim.x)
		s_b[0][l_c] = d_rii[l_c + l_r * d_riip];	
	int l_i = 0;
	for (int l_co = 1; w - l_co > -1; l_co <<= 1) {
		__syncthreads();
		for (int l_c = threadIdx.x; l_c < w + 1; l_c += blockDim.x)
			if (l_c - l_co > -1)
				s_b[1 - l_i][l_c] = s_b[l_i][l_c] + s_b[l_i][l_c - l_co];
			else
				s_b[1 - l_i][l_c] = s_b[l_i][l_c];
		l_i = 1 - l_i;
	}
	for (int l_c = threadIdx.x; l_c < w + 1; l_c += blockDim.x)
		d_rii[l_c + l_r * d_riip] = s_b[l_i][l_c];
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