import pyzed.sl as sl
import numpy as np
import json
import socket
import cv2

# UDP configuration
# UDP_IP = "127.0.0.1"
UDP_IP = "172.16.1.54"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Screen configuration (adjust these based on your physical setup)
SCREEN_WIDTH = 3.0    # Physical screen width in meters
SCREEN_HEIGHT = 3.0   # Physical screen height in meters
SCREEN_PIXEL_WIDTH = 1920
SCREEN_PIXEL_HEIGHT = 1080

# Camera to screen transformation matrix (example values - adjust according to your setup)
CAMERA_TO_SCREEN_TRANSFORM = np.array([
    [1, 0, 0, 0],     # X axis
    [0, -1, 0, 0.85], # Y axis (flipped and offset)
    [0, 0, 1, 2.0],   # Z axis (distance from camera)
    [0, 0, 0, 1]
])

def transform_to_screen_coordinates(position_3d):
    pos_homogeneous = np.append(position_3d, 1)
    pos_screen = np.dot(CAMERA_TO_SCREEN_TRANSFORM, pos_homogeneous)[:3]
    return pos_screen

def map_3d_to_screen_pixel(position_3d):
    pos_screen = transform_to_screen_coordinates(position_3d)
    
    # Convert to screen percentages
    x_percent = (pos_screen[0] / SCREEN_WIDTH) + 0.5
    y_percent = (-pos_screen[1] / SCREEN_HEIGHT) + 0.5
    
    # Convert to pixel coordinates
    x_pixel = x_percent * SCREEN_PIXEL_WIDTH
    y_pixel = y_percent * SCREEN_PIXEL_HEIGHT
    
    # Clamp values to screen boundaries
    x_pixel = np.clip(x_pixel, 0, SCREEN_PIXEL_WIDTH - 1)
    y_pixel = np.clip(y_pixel, 0, SCREEN_PIXEL_HEIGHT - 1)
    
    return (int(x_pixel), int(y_pixel))

def main():
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
                image = sl.Mat()
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = image.get_data()
                bodies = sl.Bodies()
                zed.retrieve_bodies(bodies)

                users_data = []
                for body in bodies.body_list:
                    # Body visualization
                    bb = body.bounding_box_2d.reshape((-1, 2)).astype(np.int32)
                    cv2.polylines(img, [bb], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Modified: Show distance from screen instead of body coordinates
                    transformed_pos = transform_to_screen_coordinates(body.position)
                    distance_from_screen = transformed_pos[1]
                    body_text = f"Distance: {distance_from_screen:.2f}m"
                    text_x = bb[0][0]
                    text_y = bb[0][1] - 10 if bb[0][1] > 30 else bb[0][1] + 20
                    cv2.putText(img, body_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Hand detection and visualization (unchanged)
                    right_idx = sl.BODY_18_PARTS.RIGHT_WRIST.value
                    left_idx = sl.BODY_18_PARTS.LEFT_WRIST.value
                    
                    # Right hand
                    right_2d = body.keypoint_2d[right_idx]
                    if not np.isnan(right_2d).any():
                        rx, ry = right_2d.astype(int)
                        cv2.rectangle(img, (rx-10, ry-10), (rx+10, ry+10), (0, 0, 255), 2)
                        rh_pos = body.keypoint[right_idx]
                        rh_text = f"R: {rh_pos[0]:.2f}, {rh_pos[1]:.2f}, {rh_pos[2]:.2f}"
                        cv2.putText(img, rh_text, (rx-50, ry-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    # Left hand
                    left_2d = body.keypoint_2d[left_idx]
                    if not np.isnan(left_2d).any():
                        lx, ly = left_2d.astype(int)
                        cv2.rectangle(img, (lx-10, ly-10), (lx+10, ly+10), (255, 0, 0), 2)
                        lh_pos = body.keypoint[left_idx]
                        lh_text = f"L: {lh_pos[0]:.2f}, {lh_pos[1]:.2f}, {lh_pos[2]:.2f}"
                        cv2.putText(img, lh_text, (lx+10, ly-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    # Prepare data for UDP (unchanged)
                    right_hand_3d = body.keypoint[right_idx]
                    left_hand_3d = body.keypoint[left_idx]

                    right_hand_data = None
                    if not np.isnan(right_hand_3d).any():
                        right_hand_pixel = map_3d_to_screen_pixel(right_hand_3d)
                        right_hand_data = {
                            "x": int(right_hand_pixel[0]),
                            "y": int(right_hand_pixel[1]),
                            "z": round(float(right_hand_3d[2]), 2)
                        }

                    left_hand_data = None
                    if not np.isnan(left_hand_3d).any():
                        left_hand_pixel = map_3d_to_screen_pixel(left_hand_3d)
                        left_hand_data = {
                            "x": int(left_hand_pixel[0]),
                            "y": int(left_hand_pixel[1]),
                            "z": round(float(left_hand_3d[2]), 2)
                        }

                    user_data = {
                        "person_Id": str(body.id),
                        "distance_from_screen": round(float(transformed_pos[2]), 2),
                        "right_hand": right_hand_data,
                        "left_hand": left_hand_data
                    }
                    users_data.append(user_data)

                if users_data:
                    sock.sendto(json.dumps(users_data).encode('utf-8'), (UDP_IP, UDP_PORT))

                cv2.imshow("ZED Tracking", img)
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