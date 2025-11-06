import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,  # Very low to detect even partial hands
    min_tracking_confidence=0.3,   # Very low to maintain tracking with partial visibility
    model_complexity=1             # Use more robust model
)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS for better tracking
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
draw_color = (0, 255, 255)  # Yellow (BGR)
base_thickness = 8
min_thickness = 4
max_thickness = 16
prev_x, prev_y = None, None
prev_time = None
erase_radius = 30
color_palette = [
    (0, 255, 255),   # Yellow
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (255, 0, 255),   # Magenta
    (128, 0, 128)    # Purple
]
current_color_idx = 0
smoothing_factor = 0.5  # For line smoothing
points_buffer = []  # Store recent points for smooth curves

# Hand tracking stability
last_known_position = None  # Store last position to handle tracking loss
tracking_lost_frames = 0
max_lost_frames = 10  # Increased: keep drawing for more frames after losing tracking
last_gesture_state = None  # Remember last gesture to maintain mode

# Finger indices (tip and base/pip joint pairs)
finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
finger_pips = [2, 6, 10, 14, 18]  # Base/PIP joints

def fingers_up(hand_landmarks):
    """Return a list of which fingers are up (True/False). More lenient detection."""
    fingers = []
    
    try:
        # Thumb (check X coordinate - horizontal extension)
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
            fingers.append(True)
        else:
            fingers.append(False)
        
        # Other 4 fingers (check Y coordinate - vertical extension)
        # More lenient: use margin for detection
        margin = 0.02  # Small margin for more forgiving detection
        for i in range(1, 5):
            tip_y = hand_landmarks.landmark[finger_tips[i]].y
            pip_y = hand_landmarks.landmark[finger_pips[i]].y
            if tip_y < (pip_y + margin):
                fingers.append(True)
            else:
                fingers.append(False)
    except:
        # If any landmarks are missing, assume all fingers down
        fingers = [False] * 5
    
    return fingers

import time

def calculate_pressure(speed):
    """Calculate line thickness based on movement speed (pressure simulation)."""
    # Slower movement = thicker line (more pressure)
    # Faster movement = thinner line (less pressure)
    if speed < 5:
        return max_thickness
    elif speed > 50:
        return min_thickness
    else:
        # Inverse relationship: slow = thick, fast = thin
        normalized = (speed - 5) / 45.0
        thickness = max_thickness - (normalized * (max_thickness - min_thickness))
        return int(thickness)

def draw_smooth_line(canvas, points, color, thickness):
    """Draw smooth curve through points using interpolation."""
    if len(points) < 2:
        return
    
    # Draw smooth curve through points
    for i in range(len(points) - 1):
        x1, y1, t1 = points[i]
        x2, y2, t2 = points[i + 1]
        
        # Average thickness between two points
        avg_thickness = int((t1 + t2) / 2)
        
        # Draw line segment with variable thickness
        cv2.line(canvas, (x1, y1), (x2, y2), color, avg_thickness, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Improve image quality for better tracking
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # Improve performance
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        tracking_lost_frames = 0  # Reset lost frame counter
        
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Finger states
            fingers = fingers_up(hand_landmarks)
            thumb_up = fingers[0]
            index_up = fingers[1]
            middle_up = fingers[2]
            ring_up = fingers[3]
            pinky_up = fingers[4]

            # Index fingertip coordinates
            index_tip = hand_landmarks.landmark[8]
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Store last known position and gesture state
            last_known_position = (index_x, index_y, fingers.copy())
            
            # Determine current gesture
            current_gesture = None
            if index_up and not middle_up and not ring_up and not pinky_up:
                current_gesture = 'draw'
            elif all(fingers):
                current_gesture = 'erase'
            elif index_up and middle_up and ring_up and not pinky_up:
                current_gesture = 'color_change'
            
            last_gesture_state = current_gesture

            # --- GESTURE LOGIC ---

            # 1️⃣ Draw Mode (Only Index Finger Up)
            if index_up and not middle_up and not ring_up and not pinky_up:
                current_time = time.time()
                
                # Calculate speed and pressure
                if prev_x is not None and prev_y is not None and prev_time is not None:
                    distance = np.sqrt((index_x - prev_x)**2 + (index_y - prev_y)**2)
                    time_diff = current_time - prev_time
                    speed = distance / (time_diff + 0.001)  # Avoid division by zero
                    thickness = calculate_pressure(speed)
                    
                    # Add point to buffer with thickness
                    points_buffer.append((index_x, index_y, thickness))
                    
                    # Keep buffer size manageable (last 5 points for smoothing)
                    if len(points_buffer) > 5:
                        points_buffer.pop(0)
                    
                    # Draw smooth curve
                    draw_smooth_line(canvas, points_buffer, draw_color, thickness)
                else:
                    # First point
                    points_buffer.clear()
                    points_buffer.append((index_x, index_y, base_thickness))
                
                # Visual feedback with dynamic size
                current_thickness = points_buffer[-1][2] if points_buffer else base_thickness
                cv2.circle(frame, (index_x, index_y), current_thickness + 2, draw_color, -1)
                cv2.putText(frame, "DRAWING", (index_x + 20, index_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
                
                prev_x, prev_y = index_x, index_y
                prev_time = current_time

            # 2️⃣ Erase Mode (All Fingers Open)
            elif all(fingers):
                cv2.putText(frame, "ERASER", (index_x + 20, index_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (index_x, index_y), erase_radius, (0, 0, 255), 3)
                cv2.circle(canvas, (index_x, index_y), erase_radius, (0, 0, 0), -1)
                prev_x, prev_y = None, None
                prev_time = None
                points_buffer.clear()

            # 3️⃣ Change Color (Index + Middle + Ring Up)
            elif index_up and middle_up and ring_up and not pinky_up:
                current_color_idx = (current_color_idx + 1) % len(color_palette)
                draw_color = color_palette[current_color_idx]
                cv2.putText(frame, "COLOR CHANGED!", (w//2 - 150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, draw_color, 3)
                prev_x, prev_y = None, None
                prev_time = None
                points_buffer.clear()
                cv2.waitKey(300)  # Pause to avoid rapid color changes

            else:
                prev_x, prev_y = None, None
                prev_time = None
                points_buffer.clear()

            # Draw skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    else:
        # Hand tracking lost - use last known position for extended period
        tracking_lost_frames += 1
        
        if tracking_lost_frames <= max_lost_frames and last_known_position is not None and last_gesture_state is not None:
            # Continue using last position and gesture
            index_x, index_y, fingers = last_known_position
            h, w, _ = frame.shape
            
            # Show warning indicator
            cv2.putText(frame, "TRACKING...", (w//2 - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Continue the last gesture mode
            if last_gesture_state == 'draw':
                current_time = time.time()
                
                # Calculate speed and pressure
                if prev_x is not None and prev_y is not None and prev_time is not None:
                    distance = np.sqrt((index_x - prev_x)**2 + (index_y - prev_y)**2)
                    time_diff = current_time - prev_time
                    
                    # Only draw if movement is reasonable (not too far)
                    if distance < 150:  # Prevent huge jumps
                        speed = distance / (time_diff + 0.001)
                        thickness = calculate_pressure(speed)
                        
                        points_buffer.append((index_x, index_y, thickness))
                        
                        if len(points_buffer) > 5:
                            points_buffer.pop(0)
                        
                        draw_smooth_line(canvas, points_buffer, draw_color, thickness)
                
                # Visual feedback
                current_thickness = points_buffer[-1][2] if points_buffer else base_thickness
                cv2.circle(frame, (index_x, index_y), current_thickness + 2, (100, 100, 255), -1)
                cv2.putText(frame, "DRAWING (tracking...)", (index_x + 20, index_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                
                prev_x, prev_y = index_x, index_y
                prev_time = current_time
                
            elif last_gesture_state == 'erase':
                cv2.putText(frame, "ERASER (tracking...)", (index_x + 20, index_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(frame, (index_x, index_y), erase_radius, (0, 0, 255), 2)
                cv2.circle(canvas, (index_x, index_y), erase_radius, (0, 0, 0), -1)
        else:
            # Reset drawing state after too many lost frames
            if tracking_lost_frames > max_lost_frames:
                prev_x, prev_y = None, None
                prev_time = None
                points_buffer.clear()
                last_gesture_state = None

    overlay = cv2.addWeighted(frame, 0.6, canvas, 0.8, 0)
    
    # Draw color palette indicator
    palette_x, palette_y = 20, 100
    for i, color in enumerate(color_palette):
        cv2.rectangle(overlay, (palette_x + i*50, palette_y), 
                     (palette_x + i*50 + 40, palette_y + 40), color, -1)
        if i == current_color_idx:
            cv2.rectangle(overlay, (palette_x + i*50, palette_y), 
                         (palette_x + i*50 + 40, palette_y + 40), (255, 255, 255), 3)
    
    # Instructions
    cv2.putText(overlay, "1 Finger: Draw | All Fingers: Erase | 3 Fingers: Change Color",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "Press 'C' to Clear | 'Q' to Quit",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Gesture Drawing App", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
