import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept= line_parameters       #Get the slope and intercept from left_fit and right_fit average lists
    y1= image.shape[0]                      #Get the y1 from the height of the image
    y2= int(y1*(3/5))                       #Set the y2 from 3/5 of y1
    x1= int((y1-intercept)/slope)           #Calculate x1 from y1
    x2= int((y2-intercept)/slope)           #Calculate x2 from y2
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit= []        #Coordinates of the left line
    right_fit= []       #Coordinates of the right line
    for line in lines:
        x1, y1, x2, y2= line.reshape(4)     #Detect the coordinates of each extremity of the segment lines
        parameters= np.polyfit((x1, x2), (y1, y2), 1)       #Get the slope and y intercept of the segments
        slope= parameters[0]   #Get the slope
        intercept= parameters[1]        #Get the Y-intercept
        if slope<0:
            left_fit.append((slope, intercept))     #If the slope of the line is negative the line is on the left side, then put them in the left_fit list
        else:
            right_fit.append((slope, intercept))        #If the slope of the line is positive the line is on the right side, then put them in the right_fit list
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)     #Get the average of the left_fit list
        # print(left_fit_average, 'left')
        global left_line
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)      #Get the average of the right_fit list
        # print(right_fit_average, 'right')
        global right_line
        right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    #Convert the image to grayscale
    blur= cv2.GaussianBlur(gray, (5,5), 0)           #Apply the Gaussian filter to filter out any undesired edge
    canny=cv2.Canny(blur,50,150)                     #Apply the Gradient on the image to detect the high changes of intensity (The parameters 50 and 150 are to set the lower and higher treshold values)
    return canny

def display_lines(image, lines):
    line_image= np.zeros_like(image)      #create a black image where we'll trace our lines
    if lines is not None:                 #Detect if lines are detected in gradient image
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)     #Draw the lines on the black image created with BGR values (blue line) and the thickness value of the lines
    return line_image

def region_of_interest(image):
    height= image.shape[0]           #Take the height of the image as the image is an array of 2 index values (height, width)
    width= image.shape[1]
    polygons= np.array([
    [(0, height-100), (width, height-100), (width/2, 250)]
    ])                               #Find the xy coordinates of each point of the triangle containing the region of interest (1st: left point, 2nd: right point, 3rd: top point)
    mask= np.zeros_like(image)       #create a mask image (of zeropixel values) of the same dimensions of our image
    cv2.fillPoly(mask, np.int32(polygons), 255)     #Fill our mask with the polygon in white color (255 pixel intensity)
    masked_image = cv2.bitwise_and(image, mask)     #Apply the bitwise and on the mask and the image
    return masked_image

image= cv2.imread('test_image.jpg')   # Read the image
lane_image= np.copy(image)   #import the image in a variable so that any chamge is not visible in the original image
canny_image= canny(lane_image)  #Show the gradient conversion of the original image
cropped_image = region_of_interest(canny_image)   #Show the cropped region of interest in the gradient image
lines= cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)    #1st: image where we detect lines, 2nd: ro as precision for grid in hough space, 3rd: theta for angle,4th: minimum number of votes, min line length to have for , max line gap to detect for a single line
averaged_lines= average_slope_intercept(lane_image, lines)
line_image= display_lines(lane_image, averaged_lines)         #Display lines in the black image
combo_image= cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)   #Make the addition of the black image with lines and the original colored image. 0.8: weight of the original image. 1: weight of the lined image. 1: weight of the summed image
cv2.imshow('result', combo_image)  #Show the image in a window called result
cv2.waitKey(0)  #To specify the display of the image in a certain amount of milliseconds

# cap= cv2.VideoCapture("production.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image= canny(frame)
#     cropped_image = region_of_interest(canny_image)
#     lines= cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines= average_slope_intercept(frame, lines)
#     line_image= display_lines(frame, averaged_lines)
#     combo_image= cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow('result', combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
