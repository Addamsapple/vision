#define DISTORTED_IMAGE_WIDTH 1600
#define DISTORTED_IMAGE_HEIGHT 1200

#define LEFT_RECTIFIED_COLUMN 80
#define RIGHT_RECTIFIED_COLUMN 1691
#define TOP_RECTIFIED_ROW 26
#define BOTTOM_RECTIFIED_ROW 1161
#define RECTIFIED_IMAGE_WIDTH (RIGHT_RECTIFIED_COLUMN - LEFT_RECTIFIED_COLUMN + 1)
#define RECTIFIED_IMAGE_HEIGHT (BOTTOM_RECTIFIED_ROW - TOP_RECTIFIED_ROW + 1)

#define LEFT_HORIZONTAL_FOCAL_LENGTH 2283.85459622351f
#define LEFT_VERTICAL_FOCAL_LENGTH 2251.15336449465f

#define LEFT_HORIZONTAL_PRINCIPAL_POINT 825.864513741564f
#define LEFT_VERTICAL_PRINCIPAL_POINT 671.992627056719f

#define FIRST_LEFT_DISTORTION_COEFFICIENT -0.365291721681775f
#define SECOND_LEFT_DISTORTION_COEFFICIENT 0.109452594019733f

#define LEFT_HOMOGRAPHY_00 0.000437473099113827f
#define LEFT_HOMOGRAPHY_01 2.05160449524979e-06f
#define LEFT_HOMOGRAPHY_02 -0.384012185248477f
#define LEFT_HOMOGRAPHY_10 -2.02109398837429e-06f
#define LEFT_HOMOGRAPHY_11 0.000444077208608554f
#define LEFT_HOMOGRAPHY_12 -0.275086674282606f
#define LEFT_HOMOGRAPHY_20 1.01799006266184e-05f
#define LEFT_HOMOGRAPHY_21 -1.07675268513462e-24f
#define LEFT_HOMOGRAPHY_22 0.991364584807198f

#define RIGHT_HORIZONTAL_FOCAL_LENGTH 2286.56908321177f
#define RIGHT_VERTICAL_FOCAL_LENGTH 2252.51991087664f

#define RIGHT_HORIZONTAL_PRINCIPAL_POINT 817.528670248802f
#define RIGHT_VERTICAL_PRINCIPAL_POINT 574.884574672454f

#define FIRST_RIGHT_DISTORTION_COEFFICIENT -0.414841810054081f
#define SECOND_RIGHT_DISTORTION_COEFFICIENT 0.378221771047735f

#define RIGHT_HOMOGRAPHY_00 0.000437117457289663f
#define RIGHT_HOMOGRAPHY_01 2.97796602414319e-06f
#define RIGHT_HOMOGRAPHY_02 -0.40731470106726f
#define RIGHT_HOMOGRAPHY_10 -2.87070957198331e-06f
#define RIGHT_HOMOGRAPHY_11 0.000444069974258778f
#define RIGHT_HOMOGRAPHY_12 -0.27119183635419f
#define RIGHT_HOMOGRAPHY_20 2.02611748204899e-05f
#define RIGHT_HOMOGRAPHY_21 -1.32889681235033e-06f
#define RIGHT_HOMOGRAPHY_22 0.983102995106034f

#define HORIZONTAL_FOCAL_LENGTH ((LEFT_HORIZONTAL_FOCAL_LENGTH + RIGHT_HORIZONTAL_FOCAL_LENGTH) * 0.5f)
#define VERTICAL_FOCAL_LENGTH ((LEFT_VERTICAL_FOCAL_LENGTH + RIGHT_VERTICAL_FOCAL_LENGTH) * 0.5f)
#define HORIZONTAL_PRINCIPAL_POINT ((LEFT_HORIZONTAL_PRINCIPAL_POINT + RIGHT_HORIZONTAL_PRINCIPAL_POINT) * 0.5f)
#define VERTICAL_PRINCIPAL_POINT ((LEFT_VERTICAL_PRINCIPAL_POINT + RIGHT_VERTICAL_PRINCIPAL_POINT) * 0.5f)

#define BASELINE 0.121302774569993f