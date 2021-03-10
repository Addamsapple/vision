function [rectifiedImage] = rectifyRightImage(distortedImage, rectifiedImageWidth, rectifiedImageHeight, topLeftRectifiedColumn, topLeftRectifiedRow, stereoParameters)
    leftIntrinsicMatrix = stereoParameters.CameraParameters1.IntrinsicMatrix';
    rightIntrinsicMatrix = stereoParameters.CameraParameters2.IntrinsicMatrix';
    firstDistortionCoefficient = stereoParameters.CameraParameters2.RadialDistortion(1);
    secondDistortionCoefficient = stereoParameters.CameraParameters2.RadialDistortion(2);
    translationVector = (-stereoParameters.RotationOfCamera2 * stereoParameters.TranslationOfCamera2')';
    newXAxisVector = translationVector / norm(translationVector);
    newYAxisVector = [-translationVector(2), translationVector(1), 0] / norm(translationVector(1:2));
    newZAxisVector = cross(newXAxisVector, newYAxisVector);
    homography = (leftIntrinsicMatrix + rightIntrinsicMatrix) / 2 * [newXAxisVector; newYAxisVector; newZAxisVector] * stereoParameters.RotationOfCamera2 / rightIntrinsicMatrix;
    partialInverseHomography = eye(3) / rightIntrinsicMatrix / homography;
    rectifiedImage(1 : rectifiedImageHeight, 1 : rectifiedImageWidth) = 0;
    for rectifiedImageRow = 1 : rectifiedImageHeight
        for rectifiedImageColumn = 1 : rectifiedImageWidth
            undistortedCoord = partialInverseHomography * [rectifiedImageColumn - 1 + topLeftRectifiedColumn; rectifiedImageRow - 1 + topLeftRectifiedRow; 1];
            undistortedCoord = undistortedCoord / undistortedCoord(3);
            radius = sqrt(undistortedCoord(1) ^ 2 + undistortedCoord(2) ^ 2);
            distortedCoord = [undistortedCoord(1:2) * (1 + firstDistortionCoefficient * radius ^ 2 + secondDistortionCoefficient * radius ^ 4); 1];
            distortedImageCoord = rightIntrinsicMatrix * distortedCoord;
            if round(distortedImageCoord(1) + 1) >= 1 && round(distortedImageCoord(2) + 1) >= 1 && round(distortedImageCoord(1) + 1) <= size(distortedImage, 2) && round(distortedImageCoord(2) + 1) <= size(distortedImage, 1)
                rectifiedImage(rectifiedImageRow, rectifiedImageColumn) = double(distortedImage(round(distortedImageCoord(2) + 1), round(distortedImageCoord(1) + 1))) / 255;
            end
        end
    end
end