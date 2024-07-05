import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from PIL import Image
from moviepy.editor import VideoFileClip

# image = mpimg.imread('D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/test.jpg') 

def pipeline(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    def apply_smoothing(image, kernel_size = 3):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Convert to grayscale here.
    # def convert_hsv(image):
    #     return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def convert_hls(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    def select_white_yellow(image):
        converted = convert_hls(image)
        # white color mask
        lower = np.uint8([0, 180, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        #guodao
        # lower = np.uint8([  10, 150, 100])
        # upper = np.uint8([ 100, 255, 255])
        # plt.figure()
        # plt.title('white_mask')
        # plt.imshow(white_mask)

        lower = np.uint8([10, 0,   100])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(image, lower, upper)
        # plt.figure()
        # plt.title('ywllow_mask')
        # plt.imshow(yellow_mask)
        
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask) 
        masked = cv2.bitwise_and(image, image, mask = mask) 
        # plt.figure()
        # plt.title('white_yellow_mask_combine')
        # plt.imshow(masked)
        return masked 
        # return cv2.bitwise_and(image, image, mask = mask)
    
    picking_white_yellow = select_white_yellow(image)
    # plt.figure()
    # plt.title('picking_white_yellow')
    # plt.imshow(picking_white_yellow)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gaussian_image = apply_smoothing(picking_white_yellow)
    # plt.figure()
    # plt.title('gaussian_image')
    # plt.imshow(gaussian_image)
    # Call Canny Edge Detection here.
    # cannyed_image = cv2.Canny(gaussian_image, 65, 220)
    cannyed_image = cv2.Canny(gaussian_image, 20, 220)
    # plt.figure()
    # plt.title('cannyed_image')
    # plt.imshow(cannyed_image)
    #
    def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        # channel_count = img.shape[]   * channel_count/下面的
        match_mask_color = (255,)        
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    #yellow,white
    # region_of_interest_vertices = [
    #     (0, ysize),
    #     (xsize / 2, ysize *0.6),
    #     (xsize, ysize),
    # ]
    #yellow
    # region_of_interest_vertices = [
    #     (xsize*0.16, ysize),
    #     (xsize*0.46, ysize*0.59),
    #     (xsize*0.52, ysize*0.59),
    #     (xsize*0.92, ysize),
    # ]
    #white
    # region_of_interest_vertices = [
    #     (xsize*0.16, ysize),
    #     (xsize*0.46, ysize*0.59),
    #     (xsize*0.55, ysize*0.59),
    #     (xsize*0.92, ysize),
    # ]
    #challenge
    # region_of_interest_vertices = [
    #     (xsize*0.15, ysize*0.95),
    #     (xsize*0.45, ysize*0.6),
    #     (xsize*0.57, ysize*0.6),
    #     (xsize*0.9, ysize*0.95),
    # ]
    #guodao
    region_of_interest_vertices = [
        (xsize*0.05, ysize),
        (xsize*0.42, ysize*0.61),
        (xsize*0.51, ysize*0.61),
        (xsize*0.92, ysize*0.9),
    ]
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),)
    # plt.figure()
    # plt.title('cropped_image')
    # plt.imshow(cropped_image)
    # cropped_region = region_of_interest(
    #     image,
    #     np.array([region_of_interest_vertices], np.int32),)
    # plt.figure()
    # plt.title('cropped_region')
    # plt.imshow(cropped_region)
    #
    cv2.imshow('0', cropped_image)
    cv2.waitKey(1)
    def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
        # If there are no lines to draw, exit.
        if lines is None:
            return
        # Make a copy of the original image.
        img = np.copy(img)
        # Create a blank image that matches the original in size.
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )
        # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        # Merge the image with the lines onto the original.
        img = cv2.addWeighted(img, 0.95, line_img, 1.0, 0.0)
        # Return the modified image.
        return img

    #Hough
    # lines = cv2.HoughLinesP(
    #     cropped_image,
    #     rho=2,
    #     theta=np.pi / 180,
    #     threshold=30,
    #     lines=np.array([]),
    #     minLineLength=25,
    #     maxLineGap=300
    # )
    #guodao
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi / 180,
        threshold=20,
        lines=np.array([]),
        minLineLength=10,
        maxLineGap=300
    )
    line_image = draw_lines(image, lines)
    # plt.figure()
    # plt.title('line_image')
    # plt.imshow(line_image)
    # plt.show()
    
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = int(image.shape[0] * 3.25 / 5 )# <-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    draw_lines_image = draw_lines(image,[[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y]]], thickness=6,)
    
    # plt.figure()
    # plt.title('draw_lines_image')
    # plt.imshow(draw_lines_image)
    # plt.show()
    return draw_lines_image

#圖片
# white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/test_output.jpg'
# clip1 = VideoFileClip("D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/test.jpg")
# white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/test02_output.jpg'
# clip1 = VideoFileClip("D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/test02.jpg")
# 黃線
# white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/solidYellowLeft_output.mp4'
# clip1 = VideoFileClip("D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/solidYellowLeft.mp4")
#白線
# white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/solidWhiteRight_output.mp4'
# clip1 = VideoFileClip('D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/solidWhiteRight.mp4')
# #挑戰
# white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/challenge_output.mp4'
# clip1 = VideoFileClip("D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/challenge.mp4")
# #國道
white_output = 'D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/guodao_output.mp4'
clip1 = VideoFileClip("D://20240312PERCEPTION_FOR_AUTONOMOUS_CARS2 (1)//20240312PERCEPTION_FOR_AUTONOMOUS_CARS2/guodao.mp4")

white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(white_output, audio=False)

