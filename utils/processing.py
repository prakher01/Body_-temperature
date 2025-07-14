import cv2
import numpy as np
import mediapipe as mp
import json
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = gray_image[gray_image != 0]
    average_value = np.median(non_zero_pixels)
    lower_threshold = max(0, int(average_value * lower_factor))
    upper_threshold = min(255, int(average_value * upper_factor))
    mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
    updated_frame = np.zeros_like(frame)
    updated_frame[mask] = frame[mask]
    return updated_frame

def extract_rgb_signals(frame):
    face_pixels = frame[frame.sum(axis=2) > 0]
    if len(face_pixels) == 0:
        return (0, 0, 0)
    mean_r = int(np.mean(face_pixels[:, 2]))
    mean_g = int(np.mean(face_pixels[:, 1]))
    mean_b = int(np.mean(face_pixels[:, 0]))
    return (mean_r, mean_g, mean_b)

def create_face_mask_with_colors(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                               for landmark in face_landmarks.landmark], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)

    masked_face = cv2.bitwise_and(image, image, mask=mask)
    return masked_face

def skin_segmentation(face_mask):
    hsv = cv2.cvtColor(face_mask, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(face_mask, face_mask, mask=skin_mask)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_filename = os.path.join('static', 'frame.jpg')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_mask = create_face_mask_with_colors(frame)

        if face_mask is None or not np.any(face_mask):
            rgb_values = {'r': 0, 'g': 0, 'b': 0}
            estimated_temp = 0.0
        else:
            skin_segmented = skin_segmentation(face_mask)
            face = blackout_outside_dynamic_threshold(skin_segmented)
            mean_r, mean_g, mean_b = extract_rgb_signals(face)

            estimated_temp = 36 + 0.01 * mean_r - 0.005 * mean_g + 0.008 * mean_b
            print(f"Estimated body temperature: {estimated_temp:.2f} °C")

            rgb_values = {'r': mean_r, 'g': mean_g, 'b': mean_b}

        cv2.imwrite(frame_filename, frame)
        data = {
            'frame_url': '/static/frame.jpg',
            'rgb': rgb_values,
            'estimated_temperature': round(estimated_temp, 2)
        }
        yield f"data: {json.dumps(data)}\n\n"
        cv2.waitKey(30)

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)
