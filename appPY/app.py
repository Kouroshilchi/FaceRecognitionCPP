import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from torchvision import transforms
import os
import pickle
import mediapipe as mp
import threading
import faiss
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_LOADER_DIR = ROOT.parent / "utils" / "ModelLoader"
sys.path.insert(0, str(MODEL_LOADER_DIR))

from model_pytorch import load_model  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


try:

    for candidate in ['model_weights.pt', '../models/model_weights.pt' ,r"C:\Users\kuoro\Documents\GitHub\FaceRecognitionCPP\models\model_weights.pt"]:
        if os.path.exists(candidate):
            model = load_model(candidate, device)
            break
    else:
        raise FileNotFoundError("Error")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

db_path   = 'face_embeddings.pkl'
dimension = 128

known_embeddings = np.empty((0, dimension), dtype='float32')
known_names      = []
index            = faiss.IndexFlatIP(dimension)   

try:
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
            known_embeddings = data['known_embeddings']
            known_names      = data['known_names']
        if known_embeddings.size > 0:
            index.add(known_embeddings)
        print(f"Database loaded: {len(known_names)} faces")
except Exception as e:
    print(f"Error loading face database: {e}")

mp_face_detection = mp.solutions.face_detection
face_detection    = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def get_embedding(face_img):
    try:
        if face_img is None or face_img.size == 0:
            return None
        face_pil    = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(face_tensor)
        # emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def compare_embedding(emb, threshold=0.7, k=3):
    if emb is None or index.ntotal == 0:
        return ["Unknown", 0]

    emb_np = emb.cpu().numpy().astype('float32')        
    distances, indices = index.search(emb_np, k=k)
    
    # print(f"Raw distances: {distances}")  
    
    class_sims = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        name = known_names[idx]
        class_sims.setdefault(name, []).append(dist)

    best_name, best_sim = "Unknown", 0
    for name, sims in class_sims.items():
        avg = float(np.mean(sims))
        if avg > best_sim:
            best_sim, best_name = avg, name

    # print(f"best_sim: {best_sim}, best_name: {best_name}")
    
    if best_sim > threshold:
        similarity_percentage = best_sim * 100
        return [best_name, round(similarity_percentage, 2)]
    return ["Unknown", 0]

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition App")
        self.geometry("1080x1080")

        self.video_label   = tk.Label(self)
        self.video_label.pack(pady=20)

        self.status_label  = ctk.CTkLabel(self, text="Status: Idle")
        self.status_label.pack(pady=10)

        self.start_button  = ctk.CTkButton(self, text="Start Live Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button   = ctk.CTkButton(self, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.add_face_button = ctk.CTkButton(self, text="Add New Face from Folder", command=self.add_new_face)
        self.add_face_button.pack(pady=10)

        self.add_treshold  = ctk.CTkEntry(self, placeholder_text='threshold (default: 0.7)')
        self.add_treshold.pack(pady=10)

        self.set_treshold_button = ctk.CTkButton(self, text="Set Threshold", command=self.set_threshold)
        self.set_treshold_button.pack(pady=10)


        self.cap     = None
        self.running = False
        self.thread  = None
        self.tresh   = 0.7

    def start_detection(self):
        if self.running:
            return
        try:

            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
            if not self.cap.isOpened():
                raise Exception("Could not open camera.")

            self.running = True
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.status_label.configure(text="Status: Running")
            self.thread = threading.Thread(target=self.update_video, daemon=True)
            self.thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            self.status_label.configure(text="Status: Error")

    def stop_detection(self):
        if not self.running:
            return
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Status: Idle")
        self.video_label.configure(image=None)

    def update_video(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = face_detection.process(rgb)
                ih, iw, _ = frame.shape

                if res.detections:
                    for det in res.detections[:5]:
                        bb  = det.location_data.relative_bounding_box
                        x   = max(0, int(bb.xmin * iw))
                        y   = max(0, int(bb.ymin * ih))
                        w   = min(int(bb.width * iw),  iw - x)
                        h   = min(int(bb.height * ih), ih - y)

                        face_img    = frame[y:y+h, x:x+w]
                        emb         = get_embedding(face_img)
                        name, prob  = compare_embedding(emb, self.tresh)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name,          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"{prob:.1f}%",(x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            except Exception as e:
                print(f"Video error: {e}")
                break

        self.stop_detection()

    def add_new_face(self):
        name = ctk.CTkInputDialog(text="Enter the name:", title="Add Face").get_input()
        if not name:
            return

        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return

        try:
            image_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                raise Exception("No images found in the folder.")

            embeddings = []
            for img_file in image_files:
                frame = cv2.imread(os.path.join(folder_path, img_file))
                if frame is None:
                    continue

                res = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res.detections:
                    continue

                bb  = res.detections[0].location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x   = max(0, int(bb.xmin * iw))
                y   = max(0, int(bb.ymin * ih))
                w   = min(int(bb.width * iw),  iw - x)
                h   = min(int(bb.height * ih), ih - y)

                emb = get_embedding(frame[y:y+h, x:x+w])
                if emb is not None:
                    embeddings.append(emb.cpu().numpy().flatten().astype('float32'))

            if not embeddings:
                raise Exception("No faces detected in any images.")

            new_np = np.array(embeddings, dtype='float32')
            index.add(new_np)

            global known_embeddings, known_names
            known_embeddings = np.vstack([known_embeddings, new_np]) if known_embeddings.size > 0 else new_np
            known_names.extend([name] * len(embeddings))

            with open(db_path, 'wb') as f:
                pickle.dump({'known_embeddings': known_embeddings, 'known_names': known_names}, f)

            messagebox.showinfo("Success", f"'{name}' added from {len(embeddings)} images.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_threshold(self):

        self.tresh = float(self.add_treshold.get())

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = FaceRecognitionApp()
    app.mainloop()