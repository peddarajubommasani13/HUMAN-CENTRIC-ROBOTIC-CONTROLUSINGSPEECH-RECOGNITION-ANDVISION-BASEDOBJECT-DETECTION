import cv2
import numpy as np
import time
import speech_recognition as sr
from pydobot import Dobot

# ==========================================
# DOBOT CONFIG
# ==========================================

device = Dobot(port="COM3", verbose=False)
device.speed(50)

HOME_Z = 40
PICK_Z = -43

DROP_POSITIONS = {
    "RED":   (285.7, -156.2, 40),
    "GREEN": (213.8, -206.9, 40),
    "BLUE":  (132.1, -192.0, 40),
}

SAFE_MIN_X = 150
SAFE_MAX_X = 300
SAFE_MIN_Y = -60
SAFE_MAX_Y = 250

# fine tuning offsets
FINE_OFFSET_X = -3
FINE_OFFSET_Y = 30

# ==========================================
# LOAD HOMOGRAPHY
# ==========================================

H = np.load("homography.npy")

# ==========================================
# VOICE RECOGNITION
# ==========================================

def listen_command():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("\n🎤 Say: Pick Red / Pick Green / Pick Blue")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio)
        command = command.lower()

        print("You said:", command)

        if "red" in command:
            return "RED"
        elif "green" in command:
            return "GREEN"
        elif "blue" in command:
            return "BLUE"
        else:
            return None

    except:
        print("Voice not recognized")
        return None


# ==========================================
# UTILS
# ==========================================

def clamp(val, minv, maxv):
    return max(min(val, maxv), minv)


def pixel_to_robot(px, py):

    pts = np.array([[[px, py]]], dtype=np.float32)
    robot_pts = cv2.perspectiveTransform(pts, H)

    x = float(robot_pts[0][0][0]) + FINE_OFFSET_X
    y = float(robot_pts[0][0][1]) + FINE_OFFSET_Y

    x = clamp(x, SAFE_MIN_X, SAFE_MAX_X)
    y = clamp(y, SAFE_MIN_Y, SAFE_MAX_Y)

    return x, y


# ==========================================
# ROBOT PICK FUNCTION
# ==========================================

def move_robot(robot_x, robot_y, color):

    drop_x, drop_y, drop_z = DROP_POSITIONS[color]

    print(f"\nPicking {color} at {robot_x:.1f}, {robot_y:.1f}")

    try:

        device.move_to(robot_x, robot_y, HOME_Z, 0, wait=True)
        device.move_to(robot_x, robot_y, PICK_Z, 0, wait=True)

        device.suck(True)
        time.sleep(1)

        device.move_to(robot_x, robot_y, HOME_Z, 0, wait=True)

        device.move_to(drop_x, drop_y, HOME_Z, 0, wait=True)
        device.move_to(drop_x, drop_y, drop_z, 0, wait=True)

        device.suck(False)
        time.sleep(1)

        device.move_to(drop_x, drop_y, HOME_Z, 0, wait=True)

    except Exception as e:
        print("Robot error:", e)


# ==========================================
# COLOR RANGES (FIXED)
# ==========================================

color_ranges = {

    "RED": [
        (np.array([0,120,70]), np.array([10,255,255])),
        (np.array([170,120,70]), np.array([180,255,255]))
    ],

    "GREEN": [
        (np.array([35,80,80]), np.array([85,255,255]))
    ],

    "BLUE": [
        (np.array([90,80,80]), np.array([130,255,255]))
    ]
}


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed")
    exit()

print("Camera started")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    selected_color = listen_command()

    if selected_color is None:
        continue

    print("Searching for:", selected_color)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame,(640,480))
        display = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # mask only requested color
        ranges = color_ranges[selected_color]

        mask = None

        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else mask + m

        # morphology
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 2000:
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            # avoid false detection on borders
            if x < 80:
                continue

            M = cv2.moments(cnt)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            # draw box
            cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)

            # centroid
            cv2.circle(display,(cx,cy),6,(0,0,255),-1)

            # crosshair
            cv2.line(display,(cx-10,cy),(cx+10,cy),(255,255,255),2)
            cv2.line(display,(cx,cy-10),(cx,cy+10),(255,255,255),2)

            # label
            cv2.putText(display,selected_color,
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

            robot_x, robot_y = pixel_to_robot(cx,cy)

            cv2.putText(display,
                        f"Robot ({robot_x:.1f},{robot_y:.1f})",
                        (x,y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,255),
                        1)

            cv2.imshow("Robot Vision", display)

            move_robot(robot_x, robot_y, selected_color)

            break

        cv2.imshow("Robot Vision", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        break


cap.release()
cv2.destroyAllWindows()
device.close()