import json
import socket
import numpy as np
import cv2
import pyzed.sl as sl

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5065

# Screen dimensions
SCREEN_WIDTH = 1000  # mm
SCREEN_HEIGHT = 1500  # mm

# Load calibration data
try:
    homography_matrix = np.load("homography_matrix.npy")
except FileNotFoundError:
    print("Using identity matrix as fallback")
    homography_matrix = np.eye(3)

def initialize_camera():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.enable_body_tracking(True)
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Camera initialization failed: " + str(err))
    return zed

def convert_to_screen_coordinates(camera_point):
    # Convert 3D camera coordinates to screen space
    homog_point = np.array([camera_point[0], camera_point[1], 1])
    screen_coord = np.dot(homography_matrix, homog_point)
    return {
        'x': (screen_coord[0] / screen_coord[2]) * SCREEN_WIDTH,
        'y': (screen_coord[1] / screen_coord[2]) * SCREEN_HEIGHT,
        'z': camera_point[2]
    }

def draw_body_info(image, body):
    # Get 2D bounding box
    bbox = body.bounding_box_2d
    top_left = (int(bbox[0][0]), int(bbox[0][1]))
    bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
    
    # Draw bounding box
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Get body position text
    text = f"ID: {body.id} | Dist: {body.position[2]:.2f}m"
    
    # Put text above bounding box
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw hands
    left_hand = body.keypoint[sl.BODY_PARTS.LEFT_HAND.value]
    right_hand = body.keypoint[sl.BODY_PARTS.RIGHT_HAND.value]
    
    if left_hand[0] != -1 and left_hand[1] != -1:
        cv2.circle(image, (int(left_hand[0]), int(left_hand[1])), 5, (255, 0, 0), -1)
    if right_hand[0] != -1 and right_hand[1] != -1:
        cv2.circle(image, (int(right_hand[0]), int(right_hand[1])), 5, (0, 0, 255), -1)
    
    return image

def main():
    zed = initialize_camera()
    body_tracker = sl.BodyTracker()
    bodies = sl.Bodies()
    body_runtime_params = sl.BodyTrackingRuntimeParameters()
    
    # Image containers
    image = sl.Mat()
    cv_image = np.zeros((1080, 1920, 4), dtype=np.uint8)
    
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("Starting main loop...")
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_params)
            
            # Retrieve image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            
            users = []
            
            for body in bodies.body_list:
                if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue
                
                # Draw visualization
                cv_image = draw_body_info(cv_image, body)
                
                # Prepare data for UDP
                body_position = body.position
                screen_position = convert_to_screen_coordinates(body_position)
                
                left_hand = convert_to_screen_coordinates(
                    body.keypoint[sl.BODY_PARTS.LEFT_HAND.value])
                right_hand = convert_to_screen_coordinates(
                    body.keypoint[sl.BODY_PARTS.RIGHT_HAND.value])
                
                users.append({
                    "person_Id": str(body.id),
                    "distance_from_screen": screen_position['z'],
                    "right_hand": right_hand,
                    "left_hand": left_hand
                })
            
            # Show visualization
            cv2.imshow("ZED Body Tracking", cv_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            # Send data via UDP
            users.sort(key=lambda x: x['distance_from_screen'])
            payload = json.dumps(users[:3]).encode('utf-8')
            udp_socket.sendto(payload, (UDP_IP, UDP_PORT))
    
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()