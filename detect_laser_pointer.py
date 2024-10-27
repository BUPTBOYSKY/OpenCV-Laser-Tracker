#Using python 3.7.16
import cv2
import numpy as np

#adaptive mean filter，used to remove noise while preserving image details and edges
def adaptive_mean_filter(image, kernel_size=3):
    image = image.astype(np.float32)
    mean_local = cv2.blur(image, (kernel_size, kernel_size))
    sqr_mean_local = cv2.blur(image**2, (kernel_size, kernel_size))
    variance_local = sqr_mean_local - mean_local**2
    variance_local[variance_local <= 0] = 1e-10
    variance_global = np.mean(variance_local)
    result = mean_local + (variance_local - variance_global) / variance_local * (image - mean_local)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def detect_laser_pointer():
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        camera_open = True
    else:
        camera_open = False
        print("camera wrong open")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    DISTANCE_THRESHOLD = 0.1 
    TARGET_DISTANCE = 2.0  
    brightness_threshold = 160 #Lower the brightness threshold to increase sensitivity to areas with lower brightness

    while camera_open:
        ret, frame = cap.read()
        if not ret:
            print("Image access wrong")
            break
        
        # Convert the frame from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        # Apply adaptive mean filter to the V (brightness) channel
        v_filtered = adaptive_mean_filter(v, kernel_size=3)

        _, brightness_mask = cv2.threshold(v_filtered, brightness_threshold, 255, cv2.THRESH_BINARY)

        hsv_filtered = cv2.merge([h, s, v_filtered])
        
        #Adjust the HSV threshold range to lower the lower limits of saturation and brightness
        lower_red1 = np.array([0, 120, 180])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 180])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color ranges
        mask1 = cv2.inRange(hsv_filtered, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_filtered, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_and(mask, brightness_mask)

        # Split the original frame into B, G, R channels
        b, g, r = cv2.split(frame)
        red_ratio = r.astype(float) / (g + b + 1)
        
        #Reduce the red ratio threshold and allow detection of areas with slightly lower red proportions
        _, red_mask = cv2.threshold(red_ratio, 0.5, 1.0, cv2.THRESH_BINARY)
        red_mask = (red_mask * 255).astype(np.uint8)
        mask = cv2.bitwise_and(mask, red_mask)

        # Apply erosion and dilation to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
 
        #Search for contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Sort contours in descending order of area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 1 < area < 100:  #Increase the upper limit of the area and allow detection of larger areas
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                # Calculate the circularity of the contour
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Get the bounding rectangle and calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if circularity > 0.6 and 0.7 < aspect_ratio < 1.3:
                    (x_center, y_center), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x_center), int(y_center))

                    distance = TARGET_DISTANCE  #In practical situations, distance calculation is required
                    
                    #Draw detection results
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                    cv2.putText(frame, f"Detected laser",
                                (center[0] - 30, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    #After finding the contour that meets the criteria, end the search
                    break
        
        # Display the detection result and the mask   
        cv2.imshow("Laser pointer detection", frame)
        cv2.imshow("Mask", mask)

        # Exit the loop when 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_laser_pointer()
