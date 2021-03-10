#include "cam.hpp"
#include "netc.hpp"

#include <chrono>
#include <iostream>

#include<unistd.h>

int main() {
    initializeCameras();
    connectToServer();
    for (int i = 0; i < 10; i++)
        grabFrames();
    for (int i = 0; i < 5; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        grabFrames();
        sendData(buffers[0], bufferInfo[0].bytesused);
        sendData(buffers[1], bufferInfo[1].bytesused);
        sleep(3);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "duration: " << duration << "\n";
    }
}
