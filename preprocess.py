import os
import cv2
import random
import glob
from mtcnn import MTCNN
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input

detector = MTCNN()

def extract_faces_from_video(video_path, base_output_dir, label, frame_skip=10, split_ratio=(0.7,0.15,0.15)):
    cap = cv2.VideoCapture(video_path)
    frame_no, saved = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % frame_skip == 0:
            results = detector.detect_faces(frame)
            for res in results:
                x, y, w, h = res['box']
                x, y = max(0, x), max(0, y)
                if w <= 0 or h <= 0:
                    continue
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue
           
                face = cv2.resize(face, (224,224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = preprocess_input(face.astype("float32"))

                
                if random.random() < 0.5:
                    face = cv2.flip(face, 1)

                r = random.random()
                if r < split_ratio[0]:
                    split = "train"
                elif r < split_ratio[0] + split_ratio[1]:
                    split = "val"
                else:
                    split = "test"

                out_dir = os.path.join(base_output_dir, split, label)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(out_dir, f"{label}_{os.path.basename(video_path).replace('.mp4','')}_{saved}.jpg"),
                    ((face + 1)*127.5).astype("uint8")
                )
                saved += 1
        frame_no += 1
    cap.release()


root_dir = os.getcwd()
dataset_root = os.path.join(root_dir, "dataset")

with open("List_of_testing_videos.txt", "r") as f:
    lines = f.readlines()

print("Extracting faces from videos...")
for line in tqdm(lines):
    label, path = line.strip().split(" ", 1)
    label = int(label)
    video_path = os.path.join(root_dir, path)
    if not os.path.exists(video_path):
        print(f"⚠️ Skipping {video_path}, file not found")
        continue
    if label == 1:
        extract_faces_from_video(video_path, dataset_root, "real")
    else:
        extract_faces_from_video(video_path, dataset_root, "fake")

print("Face extraction completed!")

all_imgs = glob.glob(os.path.join(dataset_root, "**", "*.jpg"), recursive=True)
count_rgb, count_gray, count_err = 0, 0, 0
for f in all_imgs:
    img = cv2.imread(f)
    if img is None:
        count_err += 1
        continue
    if len(img.shape) == 3 and img.shape[2] == 3:
        count_rgb += 1
    else:
        count_gray += 1

print(f"Total images: {len(all_imgs)}")
print(f"RGB (224x224x3): {count_rgb}")
print(f"Grayscale (224x224x1): {count_gray}")
print(f"Unreadable: {count_err}")
print("Verification complete!")
