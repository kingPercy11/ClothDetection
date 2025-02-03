import cv2
import os
import utlis


image_dir = os.path.join(os.getcwd(), 'test_data', 'u2netp_results')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("No image files found in the directory!")
else:
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading {image_path}")
            continue
        img, conts, perimeters = utlis.getContours(image, showCanny=True, minArea=50000, filter=4, draw=True)
        if perimeters:
            print(f"Perimeter of largest contour in {image_name}: {perimeters}")
        else:
            print(f"No contours found in {image_name}!")
        cv2.imshow('Frame', image)
        cv2.waitKey(0)

cv2.destroyAllWindows()