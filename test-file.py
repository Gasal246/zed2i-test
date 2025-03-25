import pyzed.sl as sl
import numpy as np
import json
import socket
import cv2
import os

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Screen dimensions (in meters and pixels)
SCREEN_WIDTH = 3.0
SCREEN_HEIGHT = 6.0
SCREEN_PIXEL_WIDTH = 1920
SCREEN_PIXEL_HEIGHT = 1080

# Touch detection threshold (in meters)
TOUCH_THRESHOLD = -1.45

# Calibration file and parameters
CALIBRATION_FILE = "calibration.npy"
calibration_points = []
homography_matrix = None

def map_3d_to_screen_pixel(position_3d):
    if homography_matrix is not None:
        src_point = np.array([position_3d[0], position_3d[1], 1])
        dst_point = np.dot(homography_matrix, src_point)
        dst_point /= dst_point[2]
        x_pixel = int(np.clip(dst_point[0], 0, SCREEN_PIXEL_WIDTH - 1))
        y_pixel = int(np.clip(dst_point[1], 0, SCREEN_PIXEL_HEIGHT - 1))
        return (x_pixel, y_pixel)
    else:
        return (0, 0)

def click_event(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
        calibration_points.append((x, y))
        print(f"Clicked point {len(calibration_points)}: ({x}, {y})")

def calibrate_camera(zed):
    global homography_matrix
    print("Starting calibration...")
    print("Click four corners of the screen in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    calibration_3d_points = []

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", click_event)

    while len(calibration_3d_points) < 4:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            img = image.get_data()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Draw existing calibration points
            for i, (px, py) in enumerate(calibration_points):
                cv2.circle(img_bgr, (px, py), 10, (0, 255, 0) if i < len(calibration_3d_points) else (0, 0, 255), 2)
                cv2.putText(img_bgr, str(i+1), (px+15, py+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # Process when we have new clicks
            while len(calibration_points) > len(calibration_3d_points):
                x, y = calibration_points[len(calibration_3d_points)]
                err, point3d = depth.get_value(x, y)
                if np.isfinite(point3d[2]):
                    calibration_3d_points.append(point3d)
                    print(f"3D point {len(calibration_3d_points)}: {point3d}")
                else:
                    print("Invalid depth at clicked point!")
                    calibration_points.pop()

            cv2.imshow("Calibration", img_bgr)
            if cv2.waitKey(10) == ord('q'):
                break

    cv2.destroyAllWindows()

    if len(calibration_3d_points) == 4:
        # Define destination points (screen corners in pixels)
        dst_points = np.array([
            [0, 0],
            [SCREEN_PIXEL_WIDTH, 0],
            [SCREEN_PIXEL_WIDTH, SCREEN_PIXEL_HEIGHT],
            [0, SCREEN_PIXEL_HEIGHT]
        ], dtype=np.float32)

        # Prepare source points (use x and z from 3D points assuming vertical screen)
        src_points = np.array([[p[0], p[2]] for p in calibration_3d_points], dtype=np.float32)

        # Calculate homography
        homography, status = cv2.findHomography(src_points, dst_points)
        if homography is not None:
            np.save(CALIBRATION_FILE, homography)
            homography_matrix = homography
            print("Calibration completed successfully!")
        else:
            print("Failed to calculate homography")

def main():
    global homography_matrix

    # Initialize ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error: Could not open ZED camera")
        exit(1)

    # Check for existing calibration
    if os.path.exists(CALIBRATION_FILE):
        homography_matrix = np.load(CALIBRATION_FILE)
        print("Loaded existing calibration")
    else:
        calibrate_camera(zed)
        if homography_matrix is None:
            print("Calibration failed, exiting")
            exit(1)

    # Rest of the tracking setup...
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    zed.enable_body_tracking(body_params)

    runtime_params = sl.RuntimeParameters()

    print("Starting tracking loop...")
    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Retrieve image and depth
                image = sl.Mat()
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = image.get_data()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Retrieve and process bodies
                bodies = sl.Bodies()
                zed.retrieve_bodies(bodies)

                bodies_list = []
                touches_list = []

                for body in bodies.body_list:
                    # Get body data
                    body_data = {
                        "person_id": str(body.id),
                        "root": {
                            "x": round(float(body.position[0]), 2),
                            "y": round(float(body.position[1]), 2),
                            "z": round(float(body.position[2]), 2)
                        },
                        "right_hand": None,
                        "left_hand": None,
                        "screen_coords": {
                            "root": None,
                            "right_hand": None,
                            "left_hand": None
                        }
                    }

                    # Map root position to screen
                    root_pixel = map_3d_to_screen_pixel(body.position)
                    body_data["screen_coords"]["root"] = {
                        "x": root_pixel[0],
                        "y": root_pixel[1]
                    }

                    # Process hands
                    right_idx = sl.BODY_18_PARTS.RIGHT_WRIST.value
                    left_idx = sl.BODY_18_PARTS.LEFT_WRIST.value

                    # Right hand
                    right_hand_3d = body.keypoint[right_idx]
                    if not np.isnan(right_hand_3d).any():
                        rh_pixel = map_3d_to_screen_pixel(right_hand_3d)
                        body_data["right_hand"] = {
                            "x": round(float(right_hand_3d[0]), 2),
                            "y": round(float(right_hand_3d[1]), 2),
                            "z": round(float(right_hand_3d[2]), 2)
                        }
                        body_data["screen_coords"]["right_hand"] = {
                            "x": rh_pixel[0],
                            "y": rh_pixel[1]
                        }

                    # Left hand
                    left_hand_3d = body.keypoint[left_idx]
                    if not np.isnan(left_hand_3d).any():
                        lh_pixel = map_3d_to_screen_pixel(left_hand_3d)
                        body_data["left_hand"] = {
                            "x": round(float(left_hand_3d[0]), 2),
                            "y": round(float(left_hand_3d[1]), 2),
                            "z": round(float(left_hand_3d[2]), 2)
                        }
                        body_data["screen_coords"]["left_hand"] = {
                            "x": lh_pixel[0],
                            "y": lh_pixel[1]
                        }

                    bodies_list.append(body_data)

                    # Touch detection
                    if body_data["right_hand"] and body_data["right_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({"person_id": str(body.id), "hand": "right_hand"})
                    if body_data["left_hand"] and body_data["left_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({"person_id": str(body.id), "hand": "left_hand"})

                # Send data via UDP
                if bodies_list:
                    data = {
                        "bodies": bodies_list,
                        "touches": touches_list
                    }
                    sock.sendto(json.dumps(data).encode('utf-8'), (UDP_IP, UDP_PORT))

                # Display image
                cv2.imshow("ZED Tracking", img_bgr)
                if cv2.waitKey(1) == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
