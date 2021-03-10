#include <glad.h>
#include <glfw3.h>

#include "mod.hpp"
#include "pro.hpp"
#include "sce.hpp"
#include "sha.hpp"

#define TITLE "PROG"
#define WIDTH 800
#define HEIGHT 800

GLFWwindow *window;

#define ROTATION_SENSITIVITY 0.003f
#define TRANSLATION_SENSITIVITY 0.02f

double xCursorPos = 0;
double yCursorPos = 0;

std::vector<Program *> programs;

unsigned int vertexArrayID;
unsigned int vertexBufferIDs[2];
unsigned int elementBufferIDs[1];

void resizeWindow(GLFWwindow* window, int width, int height);
void initializeGLFW();
void loadModels();
void initializeOpenGL();
void terminateOpenGL();
void rotateCam();
void translateCam();
void renderModel();

void renderScene() {
	initializeGLFW();
	initializeScene();
	loadModels();
	initializeOpenGL();
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		rotateCam();
		translateCam();
		transformCamera();
		renderModel();
	}
	terminateOpenGL();
	terminateScene();
	glfwTerminate();
}

void resizeWindow(GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

void initializeGLFW() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, resizeWindow);
}

void loadModels() {
	//addModel(new PointCloud("mod.p"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	addModel(new PointCloud("bruh.p"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

#include <iostream>

void initializeOpenGL() {
	gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
	glViewport(0, 0, WIDTH, HEIGHT);
	programs.push_back(new Program());
	programs[0]->add(new Shader("mesh.v", VERTEX_SHADER));
	programs[0]->add(new Shader("mesh.f", FRAGMENT_SHADER));
	programs[0]->compile();
	programs[0]->use();
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	glGenBuffers(models.size() * 2, vertexBufferIDs);
	glGenBuffers(models.size(), elementBufferIDs);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferIDs[0]);
	glBufferData(GL_ARRAY_BUFFER, models[0]->pointCount * sizeof(float) * 3, models[0]->points, GL_STATIC_DRAW);
	int vertexPositionAttribute = glGetAttribLocation(programs[0]->id, "vertexPosition");
	glVertexAttribPointer(vertexPositionAttribute, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
	glEnableVertexAttribArray(vertexPositionAttribute);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferIDs[1]);
	glBufferData(GL_ARRAY_BUFFER, models[0]->pointCount * sizeof(unsigned char) * 3, models[0]->colours, GL_STATIC_DRAW);
	int vertexColourAttribute = glGetAttribLocation(programs[0]->id, "vertexColour");
	glVertexAttribPointer(vertexColourAttribute, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(unsigned char) * 3, 0);
	glEnableVertexAttribArray(vertexColourAttribute);
	int objectTransformationUniform = glGetUniformLocation(programs[0]->id, "objectTransformation");
	glUniformMatrix4fv(objectTransformationUniform, 1, GL_FALSE, transformations[0]);
	glEnable(GL_DEPTH_TEST);
	glLineWidth(2.0f);

}

void terminateOpenGL() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(models.size() * 2, vertexBufferIDs);
	glDeleteBuffers(models.size(), elementBufferIDs);
	for (int index = programs.size() - 1; index > - 1; index--) {
		delete programs[index];
		programs.erase(programs.begin() + index);
	}
}

void rotateCam() {
	double xCursorPosN, yCursorPosN;
	glfwGetCursorPos(window, &xCursorPosN, &yCursorPosN);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		rotateCamera((xCursorPos - xCursorPosN) * ROTATION_SENSITIVITY, (yCursorPos - yCursorPosN) * ROTATION_SENSITIVITY);
	xCursorPos = xCursorPosN;
	yCursorPos = yCursorPosN;
}

void translateCam() {
	float dSway = 0.0f, dHeave = 0.0f, dSurge = 0.0f;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		dSway = 1.0f;
	else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		dSway = -1.0f;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		dHeave = 1.0f;
	else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		dHeave = -1.0f;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		dSurge = 1.0f;
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		dSurge = -1.0f;
	float speed = sqrt(dSway * dSway + dHeave * dHeave + dSurge * dSurge);
	if (speed > DBL_EPSILON) { 
		dSway *= TRANSLATION_SENSITIVITY / speed;
		dHeave *= TRANSLATION_SENSITIVITY / speed;
		dSurge *= TRANSLATION_SENSITIVITY / speed;
		translateCamera(dSway, dHeave, dSurge);
	}
}

void renderModel() {
	int cameraTransformationUniform = glGetUniformLocation(programs[0]->id, "cameraTransformation");
	glUniformMatrix4fv(cameraTransformationUniform, 1, GL_FALSE, transformationMatrix);
	//glBindBuffer(GL_ARRAY_BUFFER, vertexBufferIDs[0]);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glPointSize(0.5f);
	glDrawArrays(GL_POINTS, 0, models[0]->pointCount);
	glfwSwapBuffers(window);
}