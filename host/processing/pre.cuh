#include <cuda_runtime.h>

extern unsigned char *d_lri;
extern unsigned char *d_rri;

extern int *d_lrii;
extern cudaTextureObject_t t_lriii;

extern int *d_rrii;
extern cudaTextureObject_t t_rriii;

extern unsigned char *d_ltri;
extern unsigned char *d_rtri;

void initializePreprocessing();
void rectifyImages(unsigned char *l, unsigned char *r, unsigned char *o);
void integrateImages();
void transposeImages();