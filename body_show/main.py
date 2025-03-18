import pyzed.sl as sl
import cv2
import numpy as np

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    # Enable positional tracking
    positional_tracking_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(positional_tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking")
        zed.close()
        return

    # Configure body tracking with module ID=1
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_18
    body_params.instance_module_id = 1  # Unique ID

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable body tracking")
        zed.close()
        return

    # Initialize runtime parameters
    runtime_params = sl.RuntimeParameters()
    body_runtime_params = sl.BodyTrackingRuntimeParameters()
    body_runtime_params.detection_confidence_threshold = 40  # Adjust confidence

    image = sl.Mat()
    bodies = sl.Bodies()
    skeleton_pairs = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                      (1,8), (8,9), (9,10), (1,11), (11,12), (12,13),
                      (1,14), (14,15), (15,16)]

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Retrieve bodies with module ID=1
                zed.retrieve_bodies(bodies, body_runtime_params, body_params.instance_module_id)

                image_np = image.get_data()[:, :, :3].copy()
                image_np = np.ascontiguousarray(image_np)

                # Draw skeletons
                for body in bodies.body_list:
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        for pair in skeleton_pairs:
                            j1, j2 = pair
                            if body.keypoint_confidence[j1] > 0.5 and body.keypoint_confidence[j2] > 0.5:
                                pt1 = tuple(map(int, body.keypoint_2d[j1]))
                                pt2 = tuple(map(int, body.keypoint_2d[j2]))
                                cv2.line(image_np, pt1, pt2, (0, 255, 0), 2)

                cv2.imshow("Body Tracking", image_np)
                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        zed.disable_body_tracking()
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()