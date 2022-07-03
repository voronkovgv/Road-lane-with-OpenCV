import cv2
import numpy as np

def canny_filter(frame):
    r_channel = frame[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[r_channel > 220] = 255

    s_channel = frame[:, :, 2]
    binary2 = np.zeros_like(s_channel)
    binary2[s_channel > 211] = 1

    allBinary = np.zeros_like(binary)
    allBinary[((binary == 1) | (binary2 == 1))] = 255

    blurred_frame = cv2.GaussianBlur(allBinary, (5, 5), 0)

    return cv2.Canny(blurred_frame, 50, 150)

def check_lines(lines):
    if lines[0][0] <= -50:
        return False, "Keep left"
    elif lines[1][0] >= 690:
        return False, "Keep right"
    else:
        return True, "Ok"

def region_of_interest(image):
    polygons = np.array([(0, 320),
                         (550, 320),
                         (400, 155),
                         (200, 155)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (5/10))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

right_prev = []
left_prev = []
def calculate_lines(image, lines):

    left_fit = []
    right_fit = []

    global left_prev
    global right_prev

    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
            left_prev = left_line.copy()
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
            right_prev = right_line.copy()

        if (not left_fit) and (left_prev) and (right_prev):
            for i in range(len(right_line)):
                left_line[i] = left_prev[i] - (right_line[i] - right_prev[i])
            left_prev = left_line.copy()


        if (not right_fit) and (left_prev) and (right_prev):
            for i in range(len(right_line)):
                right_line[i] = right_prev[i] - (left_line[i] - left_prev[i])
            right_prev = right_line.copy()
        return np.array([left_line, right_line])

def display_lines(image, lines, ok, msg):
    lined_image = np.zeros_like(image)
    if lines is not None:
        i = 1
        for x1, y1, x2, y2 in lines:
            if i == 1:
                cv2.line(lined_image, (x1, y1), (x2, y2), (0, 255, 0), 8)
                i+=1
                p1 = [x1,y1,x2,y2]
            else:
                cv2.line(lined_image, (x1, y1), (x2, y2), (0, 255, 0), 8)
                pts = np.array([[[ p1[0], p1[1]], [p1[2], p1[3]], [x2,y2],[x1,y1]]], dtype=np.int32)
                if ok:
                    cv2.fillPoly(lined_image, pts, (202, 255, 192), lineType=8, shift=0, offset=None)
                    cv2.putText(lined_image, msg, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.fillPoly(lined_image, pts, (100, 100, 255), lineType=8, shift=0, offset=None)
                    cv2.putText(lined_image, msg, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return lined_image

def process_frame(frame):

    copied_frame = frame.copy()

    canny = canny_filter(frame)

    roi = region_of_interest(canny)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, np.array([()]), minLineLength=10, maxLineGap=5)

    a_lines = calculate_lines(frame, lines)

    ok, msg = check_lines(a_lines)

    lined_image = display_lines(copied_frame, a_lines, ok, msg)

    combined_frame = cv2.addWeighted(copied_frame, 0.9, lined_image, 0.5, 1)

    return combined_frame


video = cv2.VideoCapture('msu1.mp4')




if not video.isOpened():
    print("Error")

while video.isOpened():
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    success, frame = video.read()
    if not success:
        break
    frame = cv2.resize(frame, (640, 360))

    try:
        result = process_frame(frame)
    except:
        pass

    cv2.imshow('frame', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
         video.release()
         cv2.destroyAllWindows()
video.release()
cv2.destroyAllWindows()