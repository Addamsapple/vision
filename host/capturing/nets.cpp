#include <cstring>
#include <WS2tcpip.h>

#define PORT "42069"
#define SOCKET_FAMILY AF_INET
#define SOCKET_TYPE SOCK_STREAM
#define SOCKET_BUFFER_SIZE (256 * 1024)
#define DATA_BUFFER_SIZE (1024 * 1024 * 2)

void *networkBuffer;
int dataSize;

int clientSocket;

const int ERRORS[13] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112};

void initializeServer();
void acquireImage();

void initializeServer() {
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) exit(ERRORS[0]);
	addrinfo hint, *server;
	memset(&hint, 0, sizeof(hint));
	hint.ai_flags = AI_PASSIVE;
	hint.ai_family = SOCKET_FAMILY;
	hint.ai_socktype = SOCKET_TYPE;
	if (getaddrinfo(0, PORT, &hint, &server) != 0) exit(ERRORS[1]);
	int serverSocket = socket(server->ai_family, server->ai_socktype, server->ai_protocol);
	if (serverSocket == INVALID_SOCKET) exit(ERRORS[2]);
	int size = SOCKET_BUFFER_SIZE;
	if (setsockopt(serverSocket, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof(size)) != 0) exit(ERRORS[3]);
	if (bind(serverSocket, server->ai_addr, server->ai_addrlen) != 0) exit(ERRORS[4]);
	networkBuffer = malloc(DATA_BUFFER_SIZE);//use c++ memory allocation
	if (networkBuffer == NULL) exit(ERRORS[5]);
	if (listen(serverSocket, SOMAXCONN) != 0) exit(ERRORS[6]);
	sockaddr_storage clientInfo;
	size = sizeof(clientInfo);
	if ((clientSocket = accept(serverSocket, (sockaddr *) &clientInfo, &size)) == INVALID_SOCKET) exit(ERRORS[7]);
	if (closesocket(serverSocket) == SOCKET_ERROR) exit(ERRORS[8]);
}

#include <iostream>

void acquireImage() {
	int bytesReceived = 0;
	while (bytesReceived < sizeof(int)) {
		int bytes = recv(clientSocket, (char *) networkBuffer + bytesReceived, sizeof(int) - bytesReceived, 0);
		if (bytes == 0) exit(ERRORS[9]);
		if (bytes == SOCKET_ERROR) exit(ERRORS[10]);	
		bytesReceived += bytes;
	}
	bytesReceived = 0;
	dataSize = *((int *) networkBuffer);
	while (bytesReceived < dataSize) {
		int bytes = recv(clientSocket, (char *) networkBuffer + bytesReceived, dataSize - bytesReceived, 0);
		if (bytes == 0) exit(ERRORS[11]);
		if (bytes == SOCKET_ERROR) exit(ERRORS[12]);
		bytesReceived += bytes;
	}
}