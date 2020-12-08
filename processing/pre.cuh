extern unsigned char *d_lri;
extern unsigned char *d_rri;

extern unsigned char *d_ltri;
extern unsigned char *d_rtri;

void initializePreprocessing();
void rectifyImages(unsigned char *l, unsigned char *r, unsigned char *o);
void transposeRectifiedImages();