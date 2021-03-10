clear;

load('stereoParams.mat');

%if stereo parameters and images have been loaded correctly,
%errors will only occur if the parameters specified below do not
%result in fully contained coordinates of the rectified images,
%in which case these parameters should be changed to expand
%the bounds of the rectified images
    
%set input rectification parameters
%============================
leftRectifiedTopLeftImageColumn = -200;
leftRectifiedTopLeftImageRow = -200;
rightRectifiedTopLeftImageColumn = -200;
rightRectifiedTopLeftImageRow = -200;
rectifiedImageWidth = 2000;
rectifiedImageHeight = 2000;

%rectify the left stereo image
%===============================
leftImages = imageDatastore('Left\');
leftDistortedImage = readimage(leftImages, 1);
leftRectifiedImage = rectifyLeftImage(leftDistortedImage, rectifiedImageWidth, rectifiedImageHeight, leftRectifiedTopLeftImageColumn, leftRectifiedTopLeftImageRow, stereoParams);

%rectify the right stereo image
%==============================
rightImages = imageDatastore('Right\');
rightDistortedImage = readimage(rightImages, 1);
rightRectifiedImage = rectifyRightImage(rightDistortedImage, rectifiedImageWidth, rectifiedImageHeight, rightRectifiedTopLeftImageColumn, rightRectifiedTopLeftImageRow, stereoParams);

%combine the images
%==================
anaglyph = stereoAnaglyph(leftRectifiedImage, rightRectifiedImage);
figure;
imshow(anaglyph);

%save the images to file
%=======================
%imwrite(leftRectifiedImage, 'leftRectifiedImage.png');
%imwrite(rightRectifiedImage, 'rightRectifiedImage.png');
imwrite(anaglyph, 'stereoAnaglyph.png');

%compute rectification parameters
%================================
leftIntrinsicMatrix = stereoParams.CameraParameters1.IntrinsicMatrix';
rightIntrinsicMatrix = stereoParams.CameraParameters2.IntrinsicMatrix';
translationVector = (-stereoParams.RotationOfCamera2 * stereoParams.TranslationOfCamera2')';
newXAxisVector = translationVector / norm(translationVector);
newYAxisVector = [-translationVector(2), translationVector(1), 0] / norm(translationVector(1:2));
newZAxisVector = cross(newXAxisVector, newYAxisVector);
newIntrinsicMatrix = (leftIntrinsicMatrix + rightIntrinsicMatrix) / 2;
leftHomography = newIntrinsicMatrix * [newXAxisVector; newYAxisVector; newZAxisVector] / leftIntrinsicMatrix;
rightHomography = newIntrinsicMatrix * [newXAxisVector; newYAxisVector; newZAxisVector] * stereoParams.RotationOfCamera2 / rightIntrinsicMatrix;
partialInverseLeftHomography = eye(3) / leftIntrinsicMatrix / leftHomography;
partialInverseRightHomography = eye(3) / rightIntrinsicMatrix / rightHomography;

%inscribe a rectangle in the left rectified image
%================================================
[tlc1, tlr1, brc1, brr1] = inscribedRectangle(size(leftDistortedImage, 2), size(leftDistortedImage, 1), leftIntrinsicMatrix, stereoParams.CameraParameters1.RadialDistortion, leftHomography);
leftRectifiedImage(tlr1 + 1 - leftRectifiedTopLeftImageRow : brr1 + 1 - leftRectifiedTopLeftImageRow, tlc1 + 1 - leftRectifiedTopLeftImageColumn : tlc1 + 4 - leftRectifiedTopLeftImageColumn) = 1;
leftRectifiedImage(tlr1 + 1 - leftRectifiedTopLeftImageRow : brr1 + 1 - leftRectifiedTopLeftImageRow, brc1 + 1 - leftRectifiedTopLeftImageColumn : -1 : brc1 - 2 - leftRectifiedTopLeftImageColumn) = 1;
leftRectifiedImage(tlr1 + 1 - leftRectifiedTopLeftImageRow : tlr1 + 4 - leftRectifiedTopLeftImageRow, tlc1 + 1 - leftRectifiedTopLeftImageColumn : brc1 + 1 - leftRectifiedTopLeftImageColumn) = 1;
leftRectifiedImage(brr1 + 1 - leftRectifiedTopLeftImageRow : -1 : brr1 - 2 - leftRectifiedTopLeftImageRow, tlc1 + 1 - leftRectifiedTopLeftImageColumn : brc1 + 1 - leftRectifiedTopLeftImageColumn) = 1;
figure;
imshow(leftRectifiedImage);
imwrite(leftRectifiedImage, 'leftInscribedRectifedImage.png');

%inscribe a rectangle in the right rectified image
%=================================================
[tlc2, tlr2, brc2, brr2] = inscribedRectangle(size(leftDistortedImage, 2), size(leftDistortedImage, 1), rightIntrinsicMatrix, stereoParams.CameraParameters2.RadialDistortion, rightHomography);
rightRectifiedImage(tlr2 + 1 - rightRectifiedTopLeftImageRow : brr2 + 1 - rightRectifiedTopLeftImageRow, tlc2 + 1 - rightRectifiedTopLeftImageColumn : tlc2 + 4 - rightRectifiedTopLeftImageColumn) = 1;
rightRectifiedImage(tlr2 + 1 - rightRectifiedTopLeftImageRow : brr2 + 1 - rightRectifiedTopLeftImageRow, brc2 + 1 - rightRectifiedTopLeftImageColumn : -1 : brc2 - 2 - rightRectifiedTopLeftImageColumn) = 1;
rightRectifiedImage(tlr2 + 1 - rightRectifiedTopLeftImageRow : tlr2 + 4 - rightRectifiedTopLeftImageRow, tlc2 + 1 - rightRectifiedTopLeftImageColumn : brc2 + 1 - rightRectifiedTopLeftImageColumn) = 1;
rightRectifiedImage(brr2 + 1 - rightRectifiedTopLeftImageRow : -1 : brr2 - 2 - rightRectifiedTopLeftImageRow, tlc2 + 1 - rightRectifiedTopLeftImageColumn : brc2 + 1 - rightRectifiedTopLeftImageColumn) = 1;
figure;
imshow(rightRectifiedImage);
imwrite(rightRectifiedImage, 'rightInscribedRectifiedImage.png');

%display output rectification parameters
%==========================================
format long g;
disp('Left intrinsic matrix: ');
disp(leftIntrinsicMatrix);
disp('Right intrinsic matrix: ');
disp(rightIntrinsicMatrix);
disp('Left radial distortion coefficients: ');
disp(stereoParams.CameraParameters1.RadialDistortion);
disp('Right radial distortion coefficients: ');
disp(stereoParams.CameraParameters2.RadialDistortion);
disp('Partial left inverse homography: ');
disp(partialInverseLeftHomography);
disp('Partial right inverse homography: ');
disp(partialInverseRightHomography);
disp('Rectified top left image column: ');
disp(max(tlc1, tlc2));
disp('Rectified top left image row: ');
disp(max(tlr1, tlr2));
disp('Rectified bottom right image column: ');
disp(min(brc1, brc2));
disp('Rectified bottom right image row: ');
disp(min(brr1, brr2));
disp('Baseline: ');
disp(norm(translationVector) / 1000);