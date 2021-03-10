#include <linux/videodev2.h>

#define CAMERAS 2

extern void *buffers[CAMERAS];
extern v4l2_buffer bufferInfo[CAMERAS];

void initializeCameras();
void grabFrames();
