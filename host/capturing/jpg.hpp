extern unsigned char *leftJPG;
extern unsigned char *rightJPG;

extern unsigned char *leftGray;
extern unsigned char *rightGray;

void initializeJPGDecompression(unsigned char *buffer, int size);
void decompressImage(unsigned char *buffer);
void makeGray();
void initJpegMem();