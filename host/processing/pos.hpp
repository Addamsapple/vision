void performRandomSampleConsensus(float *orientation, float *position, float *pixels, float *points, int features);

void combineVectors(float *vectorA, float *vectorB, float *vectorC, float b, float c, int size);
float vectorDotProduct(float *vectorA, float *vectorB, int size);
float matrixInnerProduct(float *matrixA, float *matrixB, int size);