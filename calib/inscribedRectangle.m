function [rectifiedTopLeftColumn, rectifiedTopLeftRow, rectifiedBottomRightColumn, rectifiedBottomRightRow] = inscribedRectangle(distortedImageWidth, distortedImageHeight, intrinsicMatrix, radialDistortion, homography)
    rectifiedImageCentreCoord = homography * intrinsicMatrix * [0; 0; 1];
    rectifiedImageCentreCoord = rectifiedImageCentreCoord / rectifiedImageCentreCoord(3);
    rectifiedTopLeftColumn = round(rectifiedImageCentreCoord(1));
    rectifiedTopLeftRow = round(rectifiedImageCentreCoord(2));
    rectifiedBottomRightColumn = round(rectifiedImageCentreCoord(1));
    rectifiedBottomRightRow = round(rectifiedImageCentreCoord(2));
    partialInverseHomography = eye(3) / intrinsicMatrix / homography;
    changed = true;
    while changed
        changed = false;
        valid = true;
        for rectifiedImageColumn = rectifiedTopLeftColumn : rectifiedBottomRightColumn
            if ~isValidPixel(distortedImageWidth, distortedImageHeight, rectifiedImageColumn, rectifiedTopLeftRow - 1, intrinsicMatrix, radialDistortion, partialInverseHomography)
                valid = false;
                break;
            end
        end
        if valid
            changed = true;
            rectifiedTopLeftRow = rectifiedTopLeftRow - 1;
        end
        valid = true;
        for rectifiedImageRow = rectifiedTopLeftRow : rectifiedBottomRightRow
            if ~isValidPixel(distortedImageWidth, distortedImageHeight, rectifiedTopLeftColumn - 1, rectifiedImageRow, intrinsicMatrix, radialDistortion, partialInverseHomography)
                valid = false;
                break;
            end
        end
        if valid
            changed = true;
            rectifiedTopLeftColumn = rectifiedTopLeftColumn - 1; 
        end
        valid = true;
        for rectifiedImageColumn = rectifiedTopLeftColumn : rectifiedBottomRightColumn
            if ~isValidPixel(distortedImageWidth, distortedImageHeight, rectifiedImageColumn, rectifiedBottomRightRow + 1, intrinsicMatrix, radialDistortion, partialInverseHomography)
                valid = false;
                break;
            end
        end
        if valid
            changed = true;
            rectifiedBottomRightRow = rectifiedBottomRightRow + 1; 
        end
        valid = true;
        for rectifiedImageRow = rectifiedTopLeftRow : rectifiedBottomRightRow
            if ~isValidPixel(distortedImageWidth, distortedImageHeight, rectifiedBottomRightColumn + 1, rectifiedImageRow, intrinsicMatrix, radialDistortion, partialInverseHomography)
                valid = false;
                break;
            end
        end
        if valid
            changed = true;
            rectifiedBottomRightColumn = rectifiedBottomRightColumn + 1; 
        end
    end
end

function [valid] = isValidPixel(distortedImageWidth, distortedImageHeight, rectifiedImageColumn, rectifiedImageRow, intrinsicMatrix, radialDistortion, partialInverseHomography)
    undistortedCoord = partialInverseHomography * [rectifiedImageColumn; rectifiedImageRow; 1];
    undistortedCoord = undistortedCoord / undistortedCoord(3);
    radius = sqrt(undistortedCoord(1) ^ 2 + undistortedCoord(2) ^ 2);
    distortedCoord = [undistortedCoord(1:2) * (1 + radialDistortion(1) * radius ^ 2 + radialDistortion(2) * radius ^ 4); 1];
    distortedImageCoord = intrinsicMatrix * distortedCoord;
    valid = true;
    if ~(distortedImageCoord(1) >= 0 && distortedImageCoord(2) >= 0 && distortedImageCoord(1) + 1 <= distortedImageWidth && distortedImageCoord(2) + 1 <= distortedImageHeight)
        valid = false;
    end
end