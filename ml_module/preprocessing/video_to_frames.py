# import cv2
# import os

# def extract_frames(video_path, output_folder):

#     os.makedirs(output_folder, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     count = 0

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
#         cv2.imwrite(frame_path, frame)

#         count += 1

#     cap.release()

#     print("Total frames:", count)


# extract_frames("input_video.mp4", "frames")

import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=10):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ✅ Skip frames (VERY IMPORTANT)
        if count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()

    print("Frames saved:", saved)