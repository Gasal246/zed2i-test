import pyzed.sl as sl
import numpy as np
import time
import math

# Initialize the ZED camera
def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        exit(1)
    return zed

# Enable body tracking
def enable_body_tracking(zed):
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.image_sync = True
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
    body_params.body_format = sl.BODY_FORMAT.BODY_38

    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error enabling body tracking: {err}")
        zed.close()
        exit(1)

# Convert 3D world coordinates to 2D screen coordinates
def world_to_screen(point_3d, screen_width=3.5, screen_height=6.0, camera_height=6.0, camera_tilt_deg=30):
    x, y, z = point_3d
    tilt_rad = math.radians(camera_tilt_deg)
    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)
    
    z_adjusted = z * cos_tilt + x * sin_tilt
    x_adjusted = -z * sin_tilt + x * cos_tilt
    
    screen_x = y + (screen_width / 2)  # [-1.75, 1.75] to [0, 3.5]
    screen_y = z_adjusted  # [0, 6]
    
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    
    return [screen_x, screen_y]

# Map screen coordinates to matrix indices
def screen_to_matrix(screen_pos, screen_width=3.5, screen_height=6.0, matrix_rows=10, matrix_cols=10):
    screen_x, screen_y = screen_pos
    col = int((screen_x / screen_width) * matrix_cols)  # Scale X to column index
    row = int(((screen_height - screen_y) / screen_height) * matrix_rows)  # Invert Y for matrix (top=0)
    col = max(0, min(col, matrix_cols - 1))
    row = max(0, min(row, matrix_rows - 1))
    return row, col

# Check if a person is facing the screen
def is_facing_screen(body):
    head_pos = body.keypoint[0]
    if np.isnan(head_pos[0]):
        return False
    return head_pos[0] > 0

# Check if a person is trying to touch the screen
def is_touching_screen(body, touch_threshold=0.2, screen_width=3.5, screen_height=6.0):
    hand_indices = [20, 21, 36, 37]
    for idx in hand_indices:
        hand_pos = body.keypoint[idx]
        if not np.isnan(hand_pos[0]):
            screen_pos = world_to_screen(hand_pos, screen_width, screen_height)
            if (abs(hand_pos[0]) < touch_threshold and
                0 <= screen_pos[0] <= screen_width and
                0 <= screen_pos[1] <= screen_height):
                return True, screen_pos
    return False, None

# Filter out walking people
class PersonTracker:
    def __init__(self):
        self.tracked_people = {}
        self.max_speed = 1.0
        self.max_distance = 2.0
        self.min_dwell_time = 1.0

    def update(self, bodies, current_time):
        current_ids = set()
        for body in bodies.body_list:
            body_id = body.id
            current_ids.add(body_id)
            head_pos = body.keypoint[0]
            if np.isnan(head_pos[0]):
                continue

            if body_id not in self.tracked_people:
                self.tracked_people[body_id] = {
                    'body': body,
                    'first_seen': current_time,
                    'last_pos': head_pos
                }
            else:
                prev_data = self.tracked_people[body_id]
                prev_pos = prev_data['last_pos']
                time_diff = current_time - prev_data['first_seen']
                
                vel_y = (head_pos[1] - prev_pos[1]) / time_diff
                vel_z = (head_pos[2] - prev_pos[2]) / time_diff
                speed = np.sqrt(vel_y**2 + vel_z**2)

                prev_data['body'] = body
                prev_data['last_pos'] = head_pos
                if time_diff > 0:
                    prev_data['speed'] = speed

        self.tracked_people = {k: v for k, v in self.tracked_people.items() if k in current_ids}

    def get_interacting_people(self, current_time):
        interacting = []
        for body_id, data in self.tracked_people.items():
            body = data['body']
            head_pos = body.keypoint[0]
            dwell_time = current_time - data['first_seen']
            speed = data.get('speed', 0)

            if (head_pos[0] < self.max_distance and
                speed < self.max_speed and
                dwell_time > self.min_dwell_time):
                interacting.append(body)
        return interacting

def main():
    zed = init_zed()
    enable_body_tracking(zed)
    tracker = PersonTracker()

    runtime_params = sl.RuntimeParameters()
    bodies = sl.Bodies()

    # Matrix settings
    matrix_rows, matrix_cols = 10, 10  # 10x10 grid (adjustable)
    screen_width, screen_height = 3.5, 6.0

    print("Running detection... Press Ctrl+C to stop.")
    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies)
                current_time = time.time()
                tracker.update(bodies, current_time)
                
                interacting_people = tracker.get_interacting_people(current_time)
                
                # Initialize screen matrix
                screen_matrix = np.zeros((matrix_rows, matrix_cols), dtype=int)
                
                for body in interacting_people:
                    head_screen_pos = world_to_screen(body.keypoint[0], screen_width, screen_height)
                    head_row, head_col = screen_to_matrix(head_screen_pos, screen_width, screen_height, matrix_rows, matrix_cols)
                    screen_matrix[head_row, head_col] = 1  # Mark head with 1
                    
                    if is_facing_screen(body):
                        print(f"Person {body.id} is facing the screen.")
                        touching, touch_pos = is_touching_screen(body)
                        if touching:
                            touch_row, touch_col = screen_to_matrix(touch_pos, screen_width, screen_height, matrix_rows, matrix_cols)
                            screen_matrix[touch_row, touch_col] = 2  # Mark hand with 2
                            print(f"Person {body.id} touching at matrix: ({touch_row}, {touch_col})")
                        else:
                            print(f"Person {body.id} not touching.")
                    else:
                        print(f"Person {body.id} not facing the screen.")
                
                # Print the screen matrix
                print("\nScreen Matrix (1=Head, 2=Hand):")
                print(screen_matrix)
                print("-" * 40)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()

if __name__ == "__main__":
    main()
