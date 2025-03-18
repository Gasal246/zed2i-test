import cv2
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

# Attempt to open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", err)
    zed.close()
    exit(1)

# Enable object detection
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True
obj_param.enable_segmentation = True
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

if obj_param.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(positional_tracking_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking failed:", err)
        zed.close()
        exit(1)

err = zed.enable_object_detection(obj_param)
if err != sl.ERROR_CODE.SUCCESS:
    print("Enable object detection failed:", err)
    zed.close()
    exit(1)

objects = sl.Objects()
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 30

cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)

# Initialize img_cv with a black image
img_cv = np.zeros((720, 1280, 3), dtype=np.uint8)

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_objects(objects, obj_runtime_param)
        
        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        img_cv = img.get_data()
        
        if objects.is_new:
            obj_arr = objects.object_list
            print("Detected", len(obj_arr), "objects")
            
            for obj in obj_arr:
                top_left = obj.bounding_box_2d[0]
                bottom_right = obj.bounding_box_2d[2]
                
                cv2.rectangle(img_cv, 
                              (int(top_left[0]), int(top_left[1])), 
                              (int(bottom_right[0]), int(bottom_right[1])),
                              (0, 255, 0), 2
                            )
                
                label = f"{obj.label} ({int(obj.confidence)}%)"
                cv2.putText(img_cv, label, (int(top_left[0]), int(top_left[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    else:
        # If grabbing fails, keep displaying the last frame (or black screen)
        pass  # Or handle the error as needed
    
    cv2.imshow("Object detection with ZED", img_cv)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.disable_object_detection()
zed.close()
cv2.destroyAllWindows()