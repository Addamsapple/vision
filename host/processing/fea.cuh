#define MAXIMUM_INTEREST_POINTS 3000

#define DESCRIPTOR_BLOCK_DIMENSIONS 4
#define DESCRIPTOR_BLOCK_SIZE 5
#define DESCRIPTOR_GRID_SIZE 4

#define DESCRIPTOR_DIMENSIONS (DESCRIPTOR_GRID_SIZE * DESCRIPTOR_GRID_SIZE * DESCRIPTOR_BLOCK_DIMENSIONS)

extern int fi;

extern float lrip[MAXIMUM_INTEREST_POINTS * 2];
extern float rrip[MAXIMUM_INTEREST_POINTS * 2];

extern float lipd[MAXIMUM_INTEREST_POINTS * DESCRIPTOR_DIMENSIONS];

extern int ipm[MAXIMUM_INTEREST_POINTS * 2];
extern int ipmc;

void initializeFeatureMatching();
void computeInterestPoints();
void matchViews();
void matchFrames();