#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sysexits.h>

#define SERVER_ADDRESS "192.168.0.108"
#define SERVER_PORT "42069"
#define SOCKET_FAMILY AF_INET
#define SOCKET_TYPE SOCK_STREAM
#define SOCKET_BUFFER_SIZE (256 * 1024)

const int errors[8] = {150, 151, 152, 153, 154, 155, 156, 157};

int clientSocket;

void connectToServer();
void sendData(void *buffer, int dataSize);

void connectToServer() {
    addrinfo hint, *server;
    memset(&hint, 0, sizeof(hint));
    hint.ai_family = SOCKET_FAMILY;
    hint.ai_socktype = SOCKET_TYPE;
    if (getaddrinfo(SERVER_ADDRESS, SERVER_PORT, &hint, &server) != 0) exit(errors[0]);
    if ((clientSocket = socket(server->ai_family, server->ai_socktype, server->ai_protocol)) == -1) exit(errors[1]);
    int value = SOCKET_BUFFER_SIZE;
    if (setsockopt(clientSocket, SOL_SOCKET, SO_SNDBUF, &value, sizeof(value)) != 0) exit(errors[2]);
    value = 1;
    if (connect(clientSocket, server->ai_addr, server->ai_addrlen) != 0) exit(errors[3]);
}

void sendData(void *buffer, int dataSize) {
    int bytesSent = 0;
    while (bytesSent < sizeof(int)) {
        int bytes = send(clientSocket, (char *) &dataSize + bytesSent, sizeof(int) - bytesSent, 0);
        if (bytes == 0) exit(errors[4]);
        if (bytes == -1) exit(errors[5]);
        bytesSent += bytes;
    }
    bytesSent = 0;
    while (bytesSent < dataSize) {
        int bytes = send(clientSocket, (char *) buffer + bytesSent, dataSize - bytesSent, 0);
        if (bytes == 0) exit(errors[6]);
        if (bytes == -1) exit(errors[7]);
        bytesSent += bytes;
    }
}
