# from mtcnn import MTCNN
# import cv2
# import os

# detector = MTCNN()

# def extract_faces(input_folder, output_folder):

#     os.makedirs(output_folder, exist_ok=True)

#     for image in os.listdir(input_folder):

#         path = os.path.join(input_folder, image)

#         img = cv2.imread(path)

#         if img is None:
#             continue

#         results = detector.detect_faces(img)

#         for result in results:

#             x, y, w, h = result["box"]

#             face = img[y:y+h, x:x+w]

#             face_path = os.path.join(output_folder, image)

#             cv2.imwrite(face_path, face)


# extract_faces("frames", "faces")

from mtcnn import MTCNN
import cv2
import os

detector = MTCNN()

def extract_faces(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    count = 0

    for image in os.listdir(input_folder):
        
        path = os.path.join(input_folder, image)
        img = cv2.imread(path)

        if img is None:
            continue

        results = detector.detect_faces(img)

        for result in results:

            x, y, w, h = result["box"]

            # ✅ Fix negative values
            x, y = max(0, x), max(0, y)

            face = img[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face_path = os.path.join(output_folder, f"face_{count}.jpg")
            cv2.imwrite(face_path, face)

            count += 1

    print("Faces extracted:", count)