# from video_to_frames import extract_frames
# from face_extraction import extract_faces

# def process_video(video_path):

#     frames_folder = "temp_frames"
#     faces_folder = "temp_faces"

#     extract_frames(video_path, frames_folder)
#     extract_faces(frames_folder, faces_folder)

#     return faces_folder





# import cv2
# import os
# from mtcnn import MTCNN

# detector = MTCNN()

# def process_video(video_path):

#     frames_folder = "frames"
#     faces_folder = "faces"

#     os.makedirs(frames_folder, exist_ok=True)
#     os.makedirs(faces_folder, exist_ok=True)

#     # ---------------- EXTRACT FRAMES ----------------
#     cap = cv2.VideoCapture(video_path)

#     count = 0
#     frame_skip = 10   # skip frames for speed

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         if count % frame_skip == 0:
#             frame_path = os.path.join(frames_folder, f"frame_{count}.jpg")
#             cv2.imwrite(frame_path, frame)

#         count += 1

#     cap.release()

#     print("Frames extracted")

#     # ---------------- EXTRACT FACES ----------------
#     face_count = 0

#     for img in os.listdir(frames_folder):

#         path = os.path.join(frames_folder, img)
#         image = cv2.imread(path)

#         if image is None:
#             continue

#         results = detector.detect_faces(image)

#         for result in results:

#             x, y, w, h = result["box"]

#             x, y = max(0, x), max(0, y)

#             face = image[y:y+h, x:x+w]

#             if face.size == 0:
#                 continue

#             face_path = os.path.join(faces_folder, f"face_{face_count}.jpg")
#             cv2.imwrite(face_path, face)

#             face_count += 1

#     print("Faces extracted:", face_count)

#     return faces_folder

#trying











import cv2
import os
import shutil
from retinaface import RetinaFace

def process_video(video_path):
    frames_folder = "frames"
    faces_folder = "faces"

    # ---------------- CLEAN OLD DATA ----------------
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    if os.path.exists(faces_folder):
        shutil.rmtree(faces_folder)

    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(faces_folder, exist_ok=True)

    # ---------------- EXTRACT FRAMES ----------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps // 3))  # roughly 3 frames per second

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame_path = os.path.join(frames_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)

        count += 1

    cap.release()
    print(f"Frames extracted: {len(os.listdir(frames_folder))}")

    # ---------------- EXTRACT FACES ----------------
    face_count = 0
    for img_name in os.listdir(frames_folder):
        img_path = os.path.join(frames_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Resize frame for faster processing
        small_image = cv2.resize(image, (640, 360))

        # Detect faces using RetinaFace
        results = RetinaFace.detect_faces(small_image)

        if isinstance(results, dict):
            for key in results:
                x1, y1, x2, y2 = results[key]['facial_area']
                x1, y1 = max(0, x1), max(0, y1)
                face = small_image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_path = os.path.join(faces_folder, f"face_{face_count}.jpg")
                cv2.imwrite(face_path, face)
                face_count += 1

    print(f"Faces extracted: {face_count}")
    return faces_folder