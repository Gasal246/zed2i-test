import pyzed.sl as sl
import numpy as np
import json
import socket
import cv2

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

CAMERA_TO_SCREEN_TRANSFORM = np.eye(4)

def transform_to_screen_coordinates(position_3d):
    pos_homogeneous = np.append(position_3d, 1)
    pos_screen = np.dot(CAMERA_TO_SCREEN_TRANSFORM, pos_homogeneous)[:3]
    return pos_screen

def map_3d_to_screen_pixel(position_3d):
    pos_screen = transform_to_screen_coordinates(position_3d)
    x_pixel = (pos_screen[0] / SCREEN_WIDTH + 0.5) * SCREEN_PIXEL_WIDTH
    y_pixel = (1 - (pos_screen[1] / SCREEN_HEIGHT + 0.5)) * SCREEN_PIXEL_HEIGHT
    x_pixel = max(0, min(SCREEN_PIXEL_WIDTH - 1, x_pixel))
    y_pixel = max(0, min(SCREEN_PIXEL_HEIGHT - 1, y_pixel))
    return (x_pixel, y_pixel)

def main():
    # Initialize ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error: Could not open ZED camera")
        exit(1)

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
                img = image.get_data()  # Shape: (height, width, 4), BGRA
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR

                depth = sl.Mat()
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_data = depth.get_data()

                # Create mask for touch threshold region (Z < TOUCH_THRESHOLD)
                mask = (depth_data < TOUCH_THRESHOLD) & np.isfinite(depth_data)

                # Create highlight image for threshold visualization
                highlight = np.zeros_like(img_bgr)  # Shape: (height, width, 3)
                highlight[mask] = [0, 255, 0]  # Green in BGR for touch zone

                # Blend highlight with original image
                img_with_highlight = cv2.addWeighted(img_bgr, 1, highlight, 0.5, 0)

                # Add text label for clarity
                cv2.putText(img_with_highlight, f"Touch zone: Z < {TOUCH_THRESHOLD:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Retrieve and process bodies
                bodies = sl.Bodies()
                zed.retrieve_bodies(bodies)

                bodies_list = []
                touches_list = []

                for body in bodies.body_list:
                    # Body visualization
                    bb = body.bounding_box_2d.reshape((-1, 2)).astype(np.int32)
                    cv2.polylines(img_with_highlight, [bb], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Show Z distance (depth)
                    body_text = f"Z: {body.position[2]:.2f}m"
                    text_x = bb[0][0]
                    text_y = bb[0][1] - 10 if bb[0][1] > 30 else bb[0][1] + 20
                    cv2.putText(img_with_highlight, body_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Hand indices
                    right_idx = sl.BODY_18_PARTS.RIGHT_WRIST.value
                    left_idx = sl.BODY_18_PARTS.LEFT_WRIST.value
                    
                    # Right hand visualization
                    right_2d = body.keypoint_2d[right_idx]
                    if not np.isnan(right_2d).any():
                        rx, ry = right_2d.astype(int)
                        cv2.rectangle(img_with_highlight, (rx-10, ry-10), (rx+10, ry+10), (0, 0, 255), 2)
                        rh_pos = body.keypoint[right_idx]
                        rh_text = f"R: {rh_pos[0]:.2f}, {rh_pos[1]:.2f}, {rh_pos[2]:.2f}"
                        cv2.putText(img_with_highlight, rh_text, (rx-50, ry-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    # Left hand visualization
                    left_2d = body.keypoint_2d[left_idx]
                    if not np.isnan(left_2d).any():
                        lx, ly = left_2d.astype(int)
                        cv2.rectangle(img_with_highlight, (lx-10, ly-10), (lx+10, ly+10), (255, 0, 0), 2)
                        lh_pos = body.keypoint[left_idx]
                        lh_text = f"L: {lh_pos[0]:.2f}, {lh_pos[1]:.2f}, {lh_pos[2]:.2f}"
                        cv2.putText(img_with_highlight, lh_text, (lx+10, ly-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    # Prepare body data
                    body_data = {
                        "person_id": str(body.id),
                        "root": {
                            "x": round(float(body.position[0]), 2),
                            "y": round(float(body.position[1]), 2),
                            "z": round(float(body.position[2]), 2)
                        },
                        "right_hand": None,
                        "left_hand": None
                    }

                    # Right hand data
                    right_hand_3d = body.keypoint[right_idx]
                    if not np.isnan(right_hand_3d).any():
                        body_data["right_hand"] = {
                            "x": round(float(right_hand_3d[0]), 2),
                            "y": round(float(right_hand_3d[1]), 2),
                            "z": round(float(right_hand_3d[2]), 2)
                        }

                    # Left hand data
                    left_hand_3d = body.keypoint[left_idx]
                    if not np.isnan(left_hand_3d).any():
                        body_data["left_hand"] = {
                            "x": round(float(left_hand_3d[0]), 2),
                            "y": round(float(left_hand_3d[1]), 2),
                            "z": round(float(left_hand_3d[2]), 2)
                        }

                    bodies_list.append(body_data)

                    # Detect touches based on Z coordinate
                    if body_data["right_hand"] and body_data["right_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({"person_id": str(body.id), "hand": "right_hand"})
                    if body_data["left_hand"] and body_data["left_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({"person_id": str(body.id), "hand": "left_hand"})

                # Send data via UDP if there are bodies detected
                if bodies_list:
                    data = {
                        "bodies": bodies_list,
                        "touches": touches_list
                    }
                    sock.sendto(json.dumps(data).encode('utf-8'), (UDP_IP, UDP_PORT))

                # Display the image with threshold and overlays
                cv2.imshow("ZED Tracking", img_with_highlight)
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