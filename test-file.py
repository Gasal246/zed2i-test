import pyzed.sl as sl
import numpy as np
import json
import socket
import cv2
import sys

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

class ScreenCalibration:
    def __init__(self):
        # Screen corner coordinates in camera 3D space (meters)
        self.screen_corners_3d = None
        # Screen corner coordinates in pixel space
        self.screen_corners_pixel = None
        # Homography matrix for transformation
        self.homography_matrix = None

    def manual_calibration(self):
        """
        Manually calibrate screen corners by entering 3D coordinates 
        and corresponding pixel coordinates.
        """
        print("Screen Calibration Process")
        print("Please enter the 3D coordinates (in meters) for each screen corner:")
        
        self.screen_corners_3d = []
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        
        for name in corner_names:
            while True:
                try:
                    print(f"\nEnter {name} corner coordinates (x y z):")
                    x = float(input("X coordinate (meters): "))
                    y = float(input("Y coordinate (meters): "))
                    z = float(input("Z coordinate (meters): "))
                    self.screen_corners_3d.append([x, y, z])
                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
        
        print("\nNow enter the corresponding pixel coordinates:")
        self.screen_corners_pixel = []
        
        for name in corner_names:
            while True:
                try:
                    print(f"\nEnter {name} corner pixel coordinates (x y):")
                    x = int(input("X pixel coordinate: "))
                    y = int(input("Y pixel coordinate: "))
                    self.screen_corners_pixel.append([x, y])
                    break
                except ValueError:
                    print("Invalid input. Please enter integer values.")
        
        # Calculate homography matrix
        self.calculate_homography()
        print("\nCalibration complete!")

    def calculate_homography(self):
        """
        Calculate homography matrix for coordinate transformation.
        """
        if len(self.screen_corners_3d) != 4 or len(self.screen_corners_pixel) != 4:
            raise ValueError("Need exactly 4 corners for homography calculation")
        
        # Convert to numpy arrays
        src_pts = np.float32(self.screen_corners_3d[:, :2])  # Use only x and y
        dst_pts = np.float32(self.screen_corners_pixel)
        
        # Calculate perspective transform
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def map_3d_to_screen_pixel(self, position_3d):
        """
        Map 3D camera coordinates to screen pixel coordinates.
        """
        if self.homography_matrix is None:
            # Fallback to default linear mapping if no calibration
            x_pixel = (position_3d[0] / SCREEN_WIDTH + 0.5) * SCREEN_PIXEL_WIDTH
            y_pixel = (1 - (position_3d[1] / SCREEN_HEIGHT + 0.5)) * SCREEN_PIXEL_HEIGHT
            x_pixel = max(0, min(SCREEN_PIXEL_WIDTH - 1, x_pixel))
            y_pixel = max(0, min(SCREEN_PIXEL_HEIGHT - 1, y_pixel))
            return (x_pixel, y_pixel)
        
        # Transform 3D point to homogeneous coordinates
        point_2d = np.array([position_3d[0], position_3d[1], 1])
        
        # Apply homography
        transformed_point = np.dot(self.homography_matrix, point_2d)
        
        # Normalize homogeneous coordinates
        x = transformed_point[0] / transformed_point[2]
        y = transformed_point[1] / transformed_point[2]
        
        # Clamp to screen boundaries
        x = max(0, min(SCREEN_PIXEL_WIDTH - 1, x))
        y = max(0, min(SCREEN_PIXEL_HEIGHT - 1, y))
        
        return (x, y)

def main():
    # Screen Calibration
    calibration = ScreenCalibration()
    
    # Manually calibrate the screen
    calibration.manual_calibration()

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
                        right_hand_pixel = calibration.map_3d_to_screen_pixel(right_hand_3d)
                        body_data["right_hand"] = {
                            "x": round(float(right_hand_3d[0]), 2),
                            "y": round(float(right_hand_3d[1]), 2),
                            "z": round(float(right_hand_3d[2]), 2),
                            "pixel_x": round(float(right_hand_pixel[0]), 2),
                            "pixel_y": round(float(right_hand_pixel[1]), 2)
                        }

                    # Left hand data
                    left_hand_3d = body.keypoint[left_idx]
                    if not np.isnan(left_hand_3d).any():
                        left_hand_pixel = calibration.map_3d_to_screen_pixel(left_hand_3d)
                        body_data["left_hand"] = {
                            "x": round(float(left_hand_3d[0]), 2),
                            "y": round(float(left_hand_3d[1]), 2),
                            "z": round(float(left_hand_3d[2]), 2),
                            "pixel_x": round(float(left_hand_pixel[0]), 2),
                            "pixel_y": round(float(left_hand_pixel[1]), 2)
                        }

                    bodies_list.append(body_data)

                    # Detect touches based on Z coordinate
                    if body_data["right_hand"] and body_data["right_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({
                            "person_id": str(body.id), 
                            "hand": "right_hand",
                            "pixel_x": body_data["right_hand"]["pixel_x"],
                            "pixel_y": body_data["right_hand"]["pixel_y"]
                        })
                    if body_data["left_hand"] and body_data["left_hand"]["z"] > TOUCH_THRESHOLD:
                        touches_list.append({
                            "person_id": str(body.id), 
                            "hand": "left_hand",
                            "pixel_x": body_data["left_hand"]["pixel_x"],
                            "pixel_y": body_data["left_hand"]["pixel_y"]
                        })

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
