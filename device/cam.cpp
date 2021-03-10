#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>

#define CAMERAS 2
#define WIDTH 1600
#define HEIGHT 1200

#define STREAM V4L2_BUF_TYPE_VIDEO_CAPTURE
#define FORMAT V4L2_PIX_FMT_MJPEG
#define MEMORY V4L2_MEMORY_MMAP

const char *cameras[CAMERAS] = {"/dev/video0", "/dev/video2"};
void *buffers[CAMERAS];
v4l2_buffer bufferInfo[CAMERAS];

int descriptors[CAMERAS];

v4l2_format format;
v4l2_requestbuffers bufferRequest;

void initializeCameras();
void initializeCamera(int camera);
void grabFrames();
void grabFrame(int camera);

void initializeCameras() {
    format.type = STREAM;
    format.fmt.pix.pixelformat = FORMAT;
    format.fmt.pix.width = WIDTH;
    format.fmt.pix.height = HEIGHT;
    bufferRequest.type = STREAM;
    bufferRequest.memory = MEMORY;
    bufferRequest.count = 1;
    for (int camera = 0; camera < CAMERAS; camera++)
        initializeCamera(camera);
}

void initializeCamera(int camera) {
    descriptors[camera] = open(cameras[camera], O_RDWR);
    ioctl(descriptors[camera], VIDIOC_S_FMT, &format);
    ioctl(descriptors[camera], VIDIOC_REQBUFS, &bufferRequest);
    memset(&bufferInfo[camera], 0, sizeof(bufferInfo[camera]));
    bufferInfo[camera].type = STREAM;
    bufferInfo[camera].memory = MEMORY;
    bufferInfo[camera].index = 0;
    ioctl(descriptors[camera], VIDIOC_QUERYBUF, &bufferInfo[camera]);
    buffers[camera] = mmap(NULL, bufferInfo[camera].length, PROT_READ | PROT_WRITE, MAP_SHARED, descriptors[camera], bufferInfo[camera].m.offset);
    ioctl(descriptors[camera], VIDIOC_STREAMON, &bufferInfo[camera].type);
}

void grabFrames() {
    std::thread threads[CAMERAS];
    for (int camera = 0; camera < CAMERAS; camera++)
        threads[camera] = std::thread(&grabFrame, camera);
    for (int camera = 0; camera < CAMERAS; camera++)
        threads[camera].join();
}

void grabFrame(int camera) {
    ioctl(descriptors[camera], VIDIOC_QBUF, &bufferInfo[camera]);
    ioctl(descriptors[camera], VIDIOC_DQBUF, &bufferInfo[camera]);
}
