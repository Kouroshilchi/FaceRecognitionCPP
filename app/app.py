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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model_path = '../models/model.pt'
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("Model file not found.")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5 , 0.5,0.5], std=[0.5,0.5,0.5]),
])

db_path = 'face_embeddings.pkl'
dimension = 128 

known_embeddings = np.empty((0, dimension), dtype='float32')
known_names = []
index = faiss.IndexFlatIP(dimension)

try:
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
            known_embeddings = data['known_embeddings']
            known_names = data['known_names']
        if known_embeddings.size > 0:
            index.add(known_embeddings)
except Exception as e:
    print(f"Error loading face database: {e}")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_embedding(face_img):
    try:
        if face_img.size == 0:
            return None
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_tensor)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def compare_embedding(emb, threshold=0.7, k=5):
    if emb is None or index.ntotal == 0:
        return ["Unknown", 0]
    
    emb_np = emb.cpu().numpy().flatten().astype('float32')
    emb_np = np.expand_dims(emb_np, axis=0)
    
    distances, indices = index.search(emb_np, k=k)
    
    if len(indices[0]) == 0 or indices[0][0] == -1:
        return ["Unknown", 0]
    
    class_similarities = {}  
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        name = known_names[idx]
        similarity = dist  
        if name in class_similarities:
            class_similarities[name].append(similarity)
        else:
            class_similarities[name] = [similarity]
    
    best_name = "Unknown"
    best_avg_sim = 0
    for name, sims in class_similarities.items():
        avg_sim = np.mean(sims)
        if avg_sim > best_avg_sim:
            best_avg_sim = avg_sim
            best_name = name
    
    if best_avg_sim > threshold:
        return [best_name, round(best_avg_sim * 100, 2)]
    else:
        return ["Unknown", 0]

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition App")
        self.geometry("1080x1080")
        
        self.video_label = tk.Label(self)
        self.video_label.pack(pady=20)
        
        self.status_label = ctk.CTkLabel(self, text="Status: Idle")
        self.status_label.pack(pady=10)
        
        self.start_button = ctk.CTkButton(self, text="Start Live Detection", command=self.start_detection)
        self.start_button.pack(pady=10)
        
        self.stop_button = ctk.CTkButton(self, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        self.add_face_button = ctk.CTkButton(self, text="Add New Face from Folder", command=self.add_new_face)
        self.add_face_button.pack(pady=10)

        self.add_treshold = ctk.CTkEntry(self , placeholder_text='treshold' )
        self.add_treshold.pack(pady= 10)


        self.cap = None
        self.running = False
        self.thread = None

    def start_detection(self):
        if not self.running:
            try:
                self.tresh = self.add_treshold.get()
                try:
                    self.tresh = float(self.tresh)
                except:
                    print('please Enter a number !!')
                    self.tresh = 0.7
                
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera.")
                self.running = True
                self.start_button.configure(state=tk.DISABLED)
                self.stop_button.configure(state=tk.NORMAL)
                self.status_label.configure(text="Status: Running")
                self.thread = threading.Thread(target=self.update_video)
                self.thread.start()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {e}")
                self.status_label.configure(text="Status: Error")

    def stop_detection(self):
        if self.running:
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
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections[:5]:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        face_img = frame[y:y+h, x:x+w]
                        emb = get_embedding(face_img)

                        name, prob = compare_embedding(emb , self.tresh)


                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"{prob:.1f}%", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            except Exception as e:
                print(f"Error in video update: {e}")
                break
        
        self.stop_detection()

    def add_new_face(self):
        name = ctk.CTkInputDialog(text="Enter the name:", title="Add Face").get_input()
        if not name:
            return
        
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            messagebox.showerror("Error", "No folder selected.")
            return
        
        try:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                raise Exception("No images found in the folder.")
            
            embeddings = []
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue 
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    detection = results.detections[0] 
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.shape[0] < 160:
                        print('maybe is not real name !!!!')
                    emb = get_embedding(face_img )
                    if emb is not None:
                        embeddings.append(emb.cpu().numpy().flatten())
            
            if not embeddings:
                raise Exception("No faces detected in any images.")
            
            new_embeddings_np = np.array(embeddings).astype('float32')
            index.add(new_embeddings_np)
            global known_embeddings, known_names
            known_embeddings = np.vstack((known_embeddings, new_embeddings_np)) if known_embeddings.size > 0 else new_embeddings_np
            known_names.extend([name] * len(embeddings))
            
            with open(db_path, 'wb') as f:
                pickle.dump({
                    'known_embeddings': known_embeddings,
                    'known_names': known_names
                }, f)
            
            messagebox.showinfo("Success", f"Face for {name} added from {len(embeddings)} images.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add face: {e}")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = FaceRecognitionApp()
    app.mainloop()
