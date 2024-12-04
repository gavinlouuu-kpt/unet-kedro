#2024/12/3
##Issues:
- Currently the CV processed images are all cropped which is inconsistent to the SAM images (original size)
- If they are first cropped we lose information also SAM pipeline needs to be updated to feed the entire image instead of just the roi
- We can first process the images, crop then find contour. This way we avoid issues with finding more than one contour that is in the other part of the images.
