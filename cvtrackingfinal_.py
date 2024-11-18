import cv2
import numpy as np
import math
from scipy.interpolate import splprep, splev


# Global Variables for HSV color ranges (for tuning using sliders)
lower_blue = np.array([100, 150, 0])   # Initial values for blue
upper_blue = np.array([140, 255, 255])
lower_green = np.array([40, 100, 50])  # Initial values for green
upper_green = np.array([90, 255, 255])
triangle_vertices = []
background = None
frame_counter = 0
# Empty callback for trackbars
def nothing(x):
    pass

# Function to create a trackbar for HSV tuning
def create_hsv_tuner(window_name):
    cv2.createTrackbar('Lower H', window_name, 0, 179, nothing)
    cv2.createTrackbar('Upper H', window_name, 179, 179, nothing)
    cv2.createTrackbar('Lower S', window_name, 0, 255, nothing)
    cv2.createTrackbar('Upper S', window_name, 255, 255, nothing)
    cv2.createTrackbar('Lower V', window_name, 0, 255, nothing)
    cv2.createTrackbar('Upper V', window_name, 255, 255, nothing)

def threshold_bw(thresher):
    cv2.createTrackbar('bw', thresher, 0, 255, nothing)
    cv2.createTrackbar('edge1', thresher, 0, 255, nothing)
    cv2.createTrackbar('edge2', thresher, 0, 255, nothing)

def get_threshold(thresher):
    thresholder = cv2.getTrackbarPos('bw', thresher)
    edge1 = cv2.getTrackbarPos('edge1', thresher)
    edge2 = cv2.getTrackbarPos('edge2', thresher)
    return thresholder, edge1, edge2

# Function to get current HSV range from trackbars
def get_hsv_range(window_name):
    lower_h = cv2.getTrackbarPos('Lower H', window_name)
    upper_h = cv2.getTrackbarPos('Upper H', window_name)
    lower_s = cv2.getTrackbarPos('Lower S', window_name)
    upper_s = cv2.getTrackbarPos('Upper S', window_name)
    lower_v = cv2.getTrackbarPos('Lower V', window_name)
    upper_v = cv2.getTrackbarPos('Upper V', window_name)
    
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    
    return lower, upper

# Function to select arena bounds
def select_arena_bounds(frame):
    print("Please select the arena bounds in the image (4 points)")

    # Create a list to store the selected points
    arena_corners = []

    # Define a callback function that will be called when the user clicks on the image
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            arena_corners.append((x, y))
            for point in arena_corners:
                cv2.circle(frame, point, 3, (0, 255, 0), -1)
            cv2.imshow('Select Points', frame)

    # Set the callback function
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', on_click)

    # Display the image and wait for user input
    while True:
        cv2.imshow('Select Points', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Check if exactly 4 points were selected
    if len(arena_corners)!= 4:
        raise ValueError("You must select exactly 4 points!")

    return arena_corners

# Function to warp the perspective using homography for the arena
def warp_arena(frame, arena_bounds):
    src_points = np.float32(arena_bounds)
    height = frame.shape[0]
    width = frame.shape[1]  # Maintain a 2:1 aspect ratio for the warped arena
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Apply homography transformation
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped, matrix

# Function to apply filters for speed (Gaussian Blur, Erosion, Dilation)
def apply_filters(frame, gaussian_size=5, erode_iter=1, dilate_iter=1):
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (gaussian_size, gaussian_size), 0)
    
    # Convert to grayscale for further processing
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Apply Erosion
    eroded = cv2.erode(gray, None, iterations=erode_iter)
    
    # Apply Dilation
    dilated = cv2.dilate(eroded, None, iterations=dilate_iter)
    cv2.imshow('da computer sees', dilated)
    return dilated

def find_triangle_centroid_orientation(image_path, threshold, edge1, edge2,blank_image):
    # Load the image and preprocess
    image = image_path
    gray = image
    if image is None:
        raise ValueError("Image not found or unable to load.")
        
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, edge1, edge2)
    cv2.imshow('binary view', thresh)
    # Find contours and identify triangles
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the shape is a triangle
        if len(approx) == 3 and cv2.arcLength(contour, True) > 10:
            # Calculate the centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                return None  # Avoid division by zero if contour area is zero
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Identify the vertex opposite the hypotenuse
            vertices = [tuple(pt[0]) for pt in approx]
            dists = [np.linalg.norm(np.array(vertices[i]) - np.array(vertices[(i+1)%3])) for i in range(3)]
            hypotenuse_index = np.argmax(dists)  # Index of the hypotenuse side
            opposite_vertex_index = (hypotenuse_index + 2) % 3  # Vertex opposite the hypotenuse
            
            # Get the opposite vertex
            opposite_vertex = vertices[opposite_vertex_index]

            # Draw the centroid-to-vertex line and extend it
            cv2.line(image, (cx, cy), opposite_vertex, (255, 0, 0), 2)  # Draw initial line
            direction_vector = np.array(opposite_vertex) - np.array([cx, cy])  # Direction vector
            extension_vector = direction_vector * 2  # Extend by doubling
            end_point = tuple(np.array([cx, cy]) + extension_vector)  # New end point
            cv2.line(image, (cx, cy), end_point, (0, 0, 255), 2)  # Draw extended line
            
            # Draw the triangle, centroid, and display
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
            cv2.drawContours(blank_image, [approx], 0, (0, 255, 0), 3)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow('Triangle Tracking', image)

            
            return (cx, cy), end_point
    return None, None

# Function to detect the ball using color thresholding (blue or green)
def detect_ball(warped_arena, lower_hsv, upper_hsv,blank_image):
    hsv = cv2.cvtColor(warped_arena, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        ball_contour = max(contours, key=cv2.contourArea)
        #x, y, w, h = cv2.boundingRect(ball_contour)
        M = cv2.moments(ball_contour)
        if (M['m00'] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])   

            # compute the radius of the contour
            radius = int(math.sqrt(M["m00"] / (2 * math.pi)))

                # draw a circle around the contour
            cv2.circle(warped_arena, (cX, cY), radius, (0, 255, 0), 2)
            cv2.circle(blank_image, (cX, cY), radius, (0, 255, 0), 2)
            # Draw contour around ball
            cv2.drawContours(warped_arena, [ball_contour], -1, (0, 0, 255), 2)  # Blue contour
            cv2.drawContours(blank_image, [ball_contour], -1, (0, 0, 255), 2)  # Blue contour
        # Return center of the ball
            return (cX,cY)
    
    return None

def predict_ball_path(ball_pos, car_pos, goal_pos):
    points = np.array([car_pos, ball_pos, goal_pos], np.float32)
    x = points[:, 0]
    y = points[:, 1]
    tck, u = splprep([x, y], s=0, k=2)  
    u_fine = np.linspace(0, 1, 50)  
    x_fine, y_fine = splev(u_fine, tck)
    curve_pts = np.vstack((x_fine, y_fine)).T.astype(np.int32)
    return curve_pts


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    _, frame = cap.read()

    #arena_bounds = select_arena_bounds(frame)
    #initial_vertices = select_triangle_vertices(frame)
    cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)  # Make the window resizable
    cv2.resizeWindow("HSV", 600, 200)
    cv2.namedWindow('thresher bangladesher', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("thresher bangladesher", 600, 200)
    cv2.namedWindow("Blank Window", cv2.WINDOW_NORMAL)


    cv2.createTrackbar('bw', 'thresher bangladesher', 0, 255, nothing)
    cv2.createTrackbar('edge1','thresher bangladesher', 0, 255, nothing)
    cv2.createTrackbar('edge2', 'thresher bangladesher', 0, 255, nothing)
    create_hsv_tuner('HSV')
    #threshold_bw('thresher bangladesher')

    height, width = frame.shape[:2]
    blank_image = np.zeros((height, width, 3), np.uint8)    
    

    #cv2.resizeWindow("Blank Window", width, height)
    #cv2.imshow("Initial Frame", frame)
    
    while True:
        _, frame = cap.read()
        #warped_arena, homography_matrix = warp_arena(frame, arena_bounds)
        filtered = apply_filters(frame)
        height, width = frame.shape[:2]
        blank_image = np.ones((height, width, 3), np.uint8)    
        threshold, edge1, edge2 = get_threshold('thresher bangladesher')
        lower_hsv, upper_hsv = get_hsv_range('HSV')
        ball_position = detect_ball(frame, lower_hsv, upper_hsv,blank_image)
        centroid, orientation = find_triangle_centroid_orientation(filtered, threshold, edge1, edge2,blank_image)
        if centroid:
            print(f"Centroid: {centroid}, Orientation: {orientation} degrees")
        if ball_position:
            print("w ball")
        
        if centroid and ball_position:
            #goal_pos = (warped_arena.shape[1] // 2, warped_arena.shape[0] - 30)
            goal_pos = (frame.shape[1] // 2, frame.shape[0] - 30)
            path_points = predict_ball_path(ball_position, centroid, goal_pos)
            cv2.polylines(frame, [path_points], False, (0, 255, 0), 2)  # Green polyline
            cv2.polylines(blank_image, [path_points], False, (0, 255, 0), 2)  # Green polyline
            
        cv2.imshow("Tracked Triangle", frame)
        cv2.imshow("Arena", frame)
        cv2.imshow("Blank Window", blank_image)
        #cv2.imshow("Warped Arena", warped_arena)
        #cv2.imshow("post processed", filtered)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()