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
        return emb
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def compare_embedding(emb, threshold=0.7, k=3):
    if emb is None or index.ntotal == 0:
        return ["Unknown", 0]

    emb_np = emb.cpu().numpy().astype('float32')        
    distances, indices = index.search(emb_np, k=k)
    
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
    
    if best_sim > threshold:
        similarity_percentage = best_sim * 100
        return [best_name, round(similarity_percentage, 2)]
    return ["Unknown", 0]

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition App")
        self.geometry("1920x1000")
        self.state('zoomed')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky="nsew")
        

        self.log_frame = ctk.CTkFrame(self.main_frame)
        self.log_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.log_frame.grid_rowconfigure(0, weight=0) 
        self.log_frame.grid_rowconfigure(1, weight=1)  
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_title = ctk.CTkLabel(
            self.log_frame,
            text="📋 Detection Log",
            font=("Arial", 16, "bold")
        )
        self.log_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.log_textbox = ctk.CTkTextbox(
            self.log_frame,
            font=("Arial", 30),
            wrap="word"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.controls_frame.grid_columnconfigure(2, weight=1)
        self.controls_frame.grid_columnconfigure(3, weight=1)

        self.status_label = ctk.CTkLabel(
            self.controls_frame,
            text="⚪ Status: Idle",
            font=("Arial", 14, "bold"),
            text_color="gray"
        )
        self.status_label.grid(row=0, column=0, columnspan=4, pady=(10, 5))

        button_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_columnconfigure(3, weight=1)
        self.start_button = ctk.CTkButton(
            button_frame,
            text="▶ Start Detection",
            command=self.start_detection,
            font=("Arial", 12),
            width=150
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="⏹ Stop Detection",
            command=self.stop_detection,
            state=tk.DISABLED,
            font=("Arial", 12),
            width=150
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.add_face_button = ctk.CTkButton(
            button_frame,
            text="📁 Add Face from Folder",
            command=self.add_new_face,
            font=("Arial", 12),
            width=180
        )
        self.add_face_button.grid(row=0, column=2, padx=5, pady=5)

        threshold_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
        threshold_frame.grid(row=0, column=3, padx=5, pady=5)
        
        self.add_treshold = ctk.CTkEntry(
            threshold_frame,
            placeholder_text='Threshold (0.7)',
            font=("Arial", 11),
            width=100
        )
        self.add_treshold.pack(side=tk.LEFT, padx=(0, 5))

        self.set_treshold_button = ctk.CTkButton(
            threshold_frame,
            text="Set",
            command=self.set_threshold,
            font=("Arial", 11),
            width=60,
        )
        self.set_treshold_button.pack(side=tk.LEFT)

        self.cap = None
        self.running = False
        self.tresh = 0.7
        
        self.update_idle()
    
    def add_log(self, message, tag="info"):
        self.log_textbox.configure(state="normal")
        
        if tag == "info":
            color = "#00FF00"  
        elif tag == "warning":
            color = "#FFFF00" 
        elif tag == "error":
            color = "#FF0000" 
        elif tag == "detection":
            color = "#00BFFF" 
        else:
            color = "#FFFFFF"  
        
        self.log_textbox.insert("end", f" {message} \n")
        
        self.log_textbox.see("end")
        
        self.log_textbox.configure(state="disabled")

    def start_detection(self):
        if self.running:
            return
        try:
            self.add_log("🚀 Starting camera...", "info")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
            if not self.cap.isOpened():
                raise Exception("Could not open camera.")

            self.running = True
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.status_label.configure(text="Status: Running")
            
            self.update_frame()

        except Exception as e:
            self.add_log(f"❌ Failed to start camera: {e}", "error")
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
        self.add_log("⏹ Detection stopped", "info")

    def update_frame(self):
        if not self.running:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
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
                        
                        if name != "Unknown" and prob > 50:
                            # self.add_log(f"✅ Recognized: {name} ({prob:.1f}%)", "detection")
                            pass

                img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
        except Exception as e:
            self.add_log(f"❌ Video error: {e}", "error")
            self.stop_detection()
            return
        
        if self.running:
            self.after(5, self.update_frame)

    def update_idle(self):
        if not self.running:
            self.after(100, self.update_idle)

    def add_new_face(self):
        self.add_log("📁 Starting add face process", "info")
        
        name = ctk.CTkInputDialog(text="Enter the name:", title="Add Face").get_input()
        if not name:
            self.add_log("❌ No name entered", "warning")
            return

        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            self.add_log("❌ No folder selected", "warning")
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

            self.add_log(f"✅ '{name}' added from {len(embeddings)} images", "info")
            messagebox.showinfo("Success", f"'{name}' added from {len(embeddings)} images.")

        except Exception as e:
            self.add_log(f"❌ Error adding face: {e}", "error")
            messagebox.showerror("Error", str(e))

    def set_threshold(self):
        try:
            self.tresh = float(self.add_treshold.get())
            self.add_log(f"✅ Threshold set to {self.tresh}", "info")
            self.status_label.configure(text=f"Status: Threshold set to {self.tresh}")
        except ValueError:
            self.add_log("❌ Invalid threshold value", "error")
            messagebox.showerror("Error", "Please enter a valid number for threshold")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = FaceRecognitionApp()
    app.mainloop()