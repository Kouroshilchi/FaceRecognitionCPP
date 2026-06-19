import cv2
import mediapipe as mp
import os
import argparse
from tqdm import tqdm 


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def crop_face(image_path, output_path):

    frame = cv2.imread(image_path)
    if frame is None:
        return False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        x = max(0, x)
        y = max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)

        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            print(f"Warning: Empty crop for {image_path}")
            return False

        if face_img.shape[0] < 150 or face_img.shape[1] < 150:
            print(f"Warning: Cropped face in {image_path} is smaller than 150x150 ({face_img.shape[1]}x{face_img.shape[0]})")
            return False

        cv2.imwrite(output_path, face_img)
        return True
    else:
        print(f"Warning: No face detected in {image_path}")
        return False

def process_dataset(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    total_images = 0
    processed_images = 0

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        total_images += len([img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))])

    with tqdm(total=total_images, desc="Processing Dataset", unit="image") as pbar:
        for cls in classes:
            input_cls_path = os.path.join(input_dir, cls)
            output_cls_path = os.path.join(output_dir, cls)

            if not os.path.exists(output_cls_path):
                os.makedirs(output_cls_path)

            for img_name in os.listdir(input_cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_img_path = os.path.join(input_cls_path, img_name)
                    output_img_path = os.path.join(output_cls_path, img_name)

                    if crop_face(input_img_path, output_img_path):
                        processed_images += 1

                    pbar.update(1)

    print(f"Processed {processed_images}/{total_images} images successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop faces from dataset images using MediaPipe and save to a new folder.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input dataset directory (e.g., train or val root)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory where cropped dataset will be saved')
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir)