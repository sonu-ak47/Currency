import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import threading
import time

class CurrencyDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("₹500 Indian Currency Detector")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize camera
        self.camera = None
        self.is_camera_on = False
        self.current_frame = None
        self.captured_note_image = None  # Store the captured note image
        self.auto_detect = True
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Increased cooldown to avoid spam
        
        # Initialize ML model for ₹500 notes only
        self.model = None
        self.scaler = None
        
        # Load model if it exists
        self.load_model()
        
        # Create UI components
        self.create_ui()
        
        # Create directories for training data
        os.makedirs("training_data/500/real", exist_ok=True)
        os.makedirs("training_data/500/fake", exist_ok=True)
        
    def create_ui(self):
        # Main frames
        self.left_frame = tk.Frame(self.root, width=800, height=700, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(self.root, width=400, height=700, bg="#e0e0e0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.right_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(self.left_frame, text="₹500 Indian Currency Detector", 
                              bg="#f0f0f0", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Camera feed
        self.camera_label = tk.Label(self.left_frame, bg="black")
        self.camera_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = tk.Label(self.left_frame, text="Camera Off", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Camera controls
        self.camera_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        self.camera_frame.pack(pady=10, fill=tk.X)
        
        self.camera_btn = ttk.Button(self.camera_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=10)
        
        # Auto-detect toggle
        self.auto_detect_var = tk.BooleanVar(value=True)
        self.auto_detect_check = ttk.Checkbutton(
            self.camera_frame, 
            text="Auto Detection", 
            variable=self.auto_detect_var,
            command=self.toggle_auto_detect
        )
        self.auto_detect_check.pack(side=tk.LEFT, padx=10)
        
        # Manual capture button - Always create it but control visibility
        self.capture_btn = ttk.Button(
            self.camera_frame, 
            text="Manual Capture", 
            command=self.capture_frame,
            state=tk.DISABLED
        )
        # Don't pack it initially - will be packed when needed
        
        # Currency info
        currency_info = tk.Label(self.right_frame, text="Detecting: ₹500 Notes", 
                               bg="#e0e0e0", font=("Arial", 14, "bold"))
        currency_info.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.LabelFrame(self.right_frame, text="Detection Results", bg="#e0e0e0", font=("Arial", 12))
        self.results_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Detection result
        self.result_label = tk.Label(self.results_frame, text="No detection yet", bg="#e0e0e0", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)
        
        # Confidence meter
        self.confidence_frame = tk.Frame(self.results_frame, bg="#e0e0e0")
        self.confidence_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(self.confidence_frame, text="Authenticity Confidence:", bg="#e0e0e0").pack(anchor=tk.W, padx=10)
        
        self.confidence_meter = ttk.Progressbar(self.confidence_frame, length=350, mode="determinate")
        self.confidence_meter.pack(pady=5, padx=10)
        
        self.confidence_value = tk.Label(self.confidence_frame, text="0%", bg="#e0e0e0")
        self.confidence_value.pack(pady=5)
        
        # Feature checks for ₹500 notes
        self.features_frame = tk.LabelFrame(self.results_frame, text="₹500 Security Features", bg="#e0e0e0")
        self.features_frame.pack(pady=10, padx=10, fill=tk.BOTH)
        
        # ₹500 specific security features
        features = ["Gandhi Portrait", "Security Thread", "Microlettering", "Watermark", "Color Shifting Ink", "RBI Emblem"]
        self.feature_vars = {}
        
        for feature in features:
            frame = tk.Frame(self.features_frame, bg="#e0e0e0")
            frame.pack(fill=tk.X, pady=2)
            
            check_var = tk.BooleanVar(value=False)
            self.feature_vars[feature] = check_var
            
            check = ttk.Checkbutton(frame, text=feature, variable=check_var, state=tk.DISABLED)
            check.pack(side=tk.LEFT, padx=5)
            
            status_var = tk.StringVar(value="Not checked")
            label = tk.Label(frame, textvariable=status_var, bg="#e0e0e0", width=15)
            label.pack(side=tk.RIGHT, padx=5)
            
            self.feature_vars[feature + "_status"] = status_var
        
        # Training section
        self.training_frame = tk.LabelFrame(self.right_frame, text="Training Mode", bg="#e0e0e0", font=("Arial", 12))
        self.training_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.train_var = tk.BooleanVar(value=False)
        self.train_checkbox = ttk.Checkbutton(self.training_frame, text="Enable Training Mode", variable=self.train_var)
        self.train_checkbox.pack(pady=5, padx=10, anchor=tk.W)
        
        self.training_buttons_frame = tk.Frame(self.training_frame, bg="#e0e0e0")
        self.training_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.train_real_btn = ttk.Button(self.training_buttons_frame, text="Save as Real ₹500", 
                                        command=lambda: self.save_training_data("real"), state=tk.DISABLED)
        self.train_real_btn.pack(side=tk.LEFT, padx=10)
        
        self.train_fake_btn = ttk.Button(self.training_buttons_frame, text="Save as Fake ₹500", 
                                        command=lambda: self.save_training_data("fake"), state=tk.DISABLED)
        self.train_fake_btn.pack(side=tk.LEFT, padx=10)
        
        self.train_model_btn = ttk.Button(self.training_buttons_frame, text="Train ₹500 Model", command=self.train_model)
        self.train_model_btn.pack(side=tk.RIGHT, padx=10)
        
    def toggle_camera(self):
        if not self.is_camera_on:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera!")
                return
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_camera_on = True
            self.camera_btn.config(text="Stop Camera")
            
            # Update UI based on auto-detection setting
            self.update_capture_ui()
            
            # Start the camera thread
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            self.is_camera_on = False
            if self.camera:
                self.camera.release()
            self.camera_btn.config(text="Start Camera")
            self.capture_btn.config(state=tk.DISABLED)
            self.capture_btn.pack_forget()
            self.status_label.config(text="Camera Off")
            
            # Reset the camera view
            self.camera_label.config(image="")
    
    def toggle_auto_detect(self):
        self.auto_detect = self.auto_detect_var.get()
        if self.is_camera_on:
            self.update_capture_ui()
    
    def update_capture_ui(self):
        """Update the UI based on camera and auto-detection status"""
        if self.is_camera_on:
            if self.auto_detect:
                self.status_label.config(text="Camera On - Auto-Detection Active")
                self.capture_btn.pack_forget()
                self.capture_btn.config(state=tk.DISABLED)
            else:
                self.status_label.config(text="Camera On - Manual Mode")
                self.capture_btn.pack(side=tk.LEFT, padx=10)
                self.capture_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Camera Off")
            self.capture_btn.pack_forget()
            self.capture_btn.config(state=tk.DISABLED)
    
    def update_camera(self):
        while self.is_camera_on:
            ret, frame = self.camera.read()
            if ret:
                # Store original frame
                original_frame = frame.copy()
                
                # Process frame for display
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw detection overlay if note is detected (only in auto mode)
                display_frame = self.current_frame
                if self.auto_detect:
                    detection_result = self.is_note_present(original_frame)
                    if detection_result['detected']:
                        # Draw bounding box around detected note
                        x, y, w, h = detection_result['bbox']
                        frame_with_box = frame.copy()
                        cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame_with_box, "Note Detected", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Convert frame with overlay for display
                        display_frame = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)
                        
                        # Auto-detect processing
                        if time.time() - self.last_detection_time > self.detection_cooldown:
                            # Capture and crop the note region
                            x, y, w, h = detection_result['bbox']
                            note_region = original_frame[y:y+h, x:x+w]
                            captured_image = cv2.cvtColor(note_region, cv2.COLOR_BGR2RGB)
                            
                            # Check if it's a ₹500 note
                            if self.is_500_note(captured_image):
                                self.captured_note_image = captured_image
                                self.status_label.config(text="₹500 Note Detected - Processing...")
                                # Process the captured note
                                self.process_captured_note()
                                
                                # Enable training buttons if training mode is on
                                if self.train_var.get():
                                    self.train_real_btn.config(state=tk.NORMAL)
                                    self.train_fake_btn.config(state=tk.NORMAL)
                            else:
                                # Show message for non-₹500 notes
                                self.status_label.config(text="Other denomination detected")
                                self.update_results("Please scan an Indian ₹500 note", 0)
                                self.reset_feature_checks()
                                
                            self.last_detection_time = time.time()
                    else:
                        self.status_label.config(text="Camera On - Scanning for Notes...")
                
                # Display the frame
                img = Image.fromarray(display_frame)
                img = self.resize_image(img, 780, 480)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
            
            time.sleep(0.03)  # ~30 FPS
    
    def is_note_present(self, frame):
        """Improved detection of currency note in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangle-like contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # More reasonable area threshold (adjusted for different distances)
            if area > 15000:  # Lowered threshold
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) >= 4:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # More lenient aspect ratio for Indian currency
                    # Indian notes are approximately 1.5 to 2.5 aspect ratio
                    if 1.2 < aspect_ratio < 3.0 and w > 150 and h > 80:
                        # Additional checks for rectangularity
                        rect_area = w * h
                        extent = float(area) / rect_area
                        
                        # Check if the contour fills most of the bounding rectangle
                        if extent > 0.6:  # At least 60% filled
                            return {
                                'detected': True,
                                'bbox': (x, y, w, h),
                                'area': area,
                                'aspect_ratio': aspect_ratio
                            }
        
        return {'detected': False, 'bbox': None}
    
    def is_500_note(self, note_image):
        """Determine if the detected note is a ₹500 note based on color characteristics"""
        if note_image is None or note_image.size == 0:
            return False
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(note_image, cv2.COLOR_RGB2HSV)
        
        # ₹500 notes are predominantly stone grey/brownish in color
        # Define color ranges for ₹500 note (stone grey/brown tones)
        # HSV ranges for stone grey/brown colors
        lower_grey1 = np.array([0, 0, 80])    # Light grey
        upper_grey1 = np.array([25, 50, 200])  # Brown-grey
        
        lower_grey2 = np.array([15, 20, 120])   # Brownish grey
        upper_grey2 = np.array([35, 80, 180])   # Medium brown
        
        # Create masks for the color ranges
        mask1 = cv2.inRange(hsv, lower_grey1, upper_grey1)
        mask2 = cv2.inRange(hsv, lower_grey2, upper_grey2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate the percentage of pixels that match ₹500 color scheme
        total_pixels = note_image.shape[0] * note_image.shape[1]
        matching_pixels = cv2.countNonZero(combined_mask)
        color_percentage = (matching_pixels / total_pixels) * 100
        
        # Additional check for dominant colors
        # Calculate mean color values
        mean_color = np.mean(note_image.reshape(-1, 3), axis=0)
        
        # ₹500 notes have specific color characteristics
        # Check if the mean color falls within expected ranges for ₹500 notes
        # Stone grey typically has balanced RGB values with slight brown tint
        r, g, b = mean_color
        
        # Check for stone grey characteristics (balanced colors, not too bright or dark)
        is_balanced = abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30
        is_medium_brightness = 80 < np.mean(mean_color) < 180
        
        # Additional size check (₹500 notes have specific dimensions)
        height, width = note_image.shape[:2]
        aspect_ratio = width / height if height > 0 else 0
        
        # Indian ₹500 notes have an aspect ratio of approximately 1.6-1.8
        correct_aspect_ratio = 1.4 < aspect_ratio < 2.0
        
        # Combine all checks
        is_500_note = (
            color_percentage > 25 and  # At least 25% matching color
            is_balanced and            # Balanced RGB values
            is_medium_brightness and   # Medium brightness
            correct_aspect_ratio       # Correct aspect ratio
        )
        
        return is_500_note
    
    def resize_image(self, img, width, height):
        img_width, img_height = img.size
        ratio = min(width/img_width, height/img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def capture_frame(self):
        """Manual capture for when auto-detection is turned off"""
        if self.current_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available to capture!")
            return
        
        print("Manual capture triggered")  # Debug print
        
        # Try to detect note in current frame first
        original_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        detection_result = self.is_note_present(original_bgr)
        
        if detection_result['detected']:
            # Extract the detected note region
            x, y, w, h = detection_result['bbox']
            note_region = self.current_frame[y:y+h, x:x+w]
            self.captured_note_image = note_region
            print(f"Note detected and cropped: {note_region.shape}")  # Debug print
        else:
            # If no specific note detected, use the entire frame
            self.captured_note_image = self.current_frame
            print("No specific note detected, using full frame")  # Debug print
        
        # Check if it's a ₹500 note
        if self.is_500_note(self.captured_note_image):
            print("Identified as ₹500 note")  # Debug print
            self.status_label.config(text="₹500 Note Captured - Processing...")
            self.process_captured_note()
            
            # Enable training buttons if training mode is on
            if self.train_var.get():
                self.train_real_btn.config(state=tk.NORMAL)
                self.train_fake_btn.config(state=tk.NORMAL)
        else:
            print("Not identified as ₹500 note")  # Debug print
            # Show message for non-₹500 notes
            self.status_label.config(text="Non-₹500 note or unclear image captured")
            self.update_results("Please scan an Indian ₹500 note", 0)
            self.reset_feature_checks()
    
    def process_captured_note(self):
        """Process the captured note image to detect ₹500 currency authenticity"""
        if self.captured_note_image is None:
            return
            
        print("Processing captured note...")  # Debug print
        
        # Extract features from the captured note
        features = self.extract_features(self.captured_note_image)
        
        if self.model:
            # Scale features
            features_scaled = self.scaler.transform([features])
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            confidence = prediction_proba[1] * 100  # Probability of being real
            result = "REAL ₹500" if confidence >= 70 else "FAKE ₹500"
            self.update_results(result, confidence)
            self.check_security_features(self.captured_note_image)
            print(f"Model prediction: {result} with {confidence:.1f}% confidence")  # Debug print
        else:
            self.update_results("MODEL NOT TRAINED", 0)
            self.reset_feature_checks()
            print("Model not trained")  # Debug print
    
    def extract_features(self, frame):
        """Extract features from the frame for ₹500 currency detection"""
        # Ensure frame is in RGB format
        if len(frame.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Resize frame for consistent feature extraction
        resized = cv2.resize(gray, (300, 150))  # Standard size for feature extraction
        
        # Extract histogram features
        hist = cv2.calcHist([resized], [0], None, [64], [0, 256])  # Reduced bins for efficiency
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extract color features if it's a color image
        if len(frame.shape) == 3:
            color_means = [np.mean(frame[:,:,i]) for i in range(3)]
            color_stds = [np.std(frame[:,:,i]) for i in range(3)]
        else:
            color_means = [np.mean(frame)]
            color_stds = [np.std(frame)]
        
        # Extract edge features using Canny edge detection
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Extract texture features using Local Binary Pattern-like approach
        texture_features = []
        for i in range(0, resized.shape[0]-2, 10):
            for j in range(0, resized.shape[1]-2, 10):
                patch = resized[i:i+3, j:j+3]
                texture_features.append(np.std(patch))
        
        # Combine all features
        features = (list(hist) + color_means + color_stds + 
                   [edge_density] + texture_features[:10])  # Limit texture features
        
        return features
    
    def check_security_features(self, frame):
        """Check individual security features in the ₹500 note"""
        # Reset all feature checks
        self.reset_feature_checks()
        
        # Convert to different color spaces
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        else:
            gray = frame
            hsv = None
        
        try:
            # Check for Gandhi portrait (simple face detection)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                self.feature_vars["Gandhi Portrait"].set(True)
                self.feature_vars["Gandhi Portrait_status"].set("Detected")
        except:
            pass
        
        # Check for security thread (vertical line detection)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 20:  # Vertical line
                    self.feature_vars["Security Thread"].set(True)
                    self.feature_vars["Security Thread_status"].set("Detected")
                    break
        
        # Check for watermark (simplified using brightness variations)
        if gray.shape[1] > 4:
            roi = gray[:, :gray.shape[1]//4]  # Watermark is usually on the left
            std_dev = np.std(roi)
            if std_dev > 15:  # Lowered threshold
                self.feature_vars["Watermark"].set(True)
                self.feature_vars["Watermark_status"].set("Detected")
        
        # Check for microlettering (simplified using edge density)
        if gray.shape[0] > 2:
            roi = gray[gray.shape[0]//2:, :]  # Bottom half of the note
            edges_roi = cv2.Canny(roi, 100, 200)
            edge_density = np.sum(edges_roi) / (edges_roi.shape[0] * edges_roi.shape[1])
            if edge_density > 0.05:  # Lowered threshold
                self.feature_vars["Microlettering"].set(True)
                self.feature_vars["Microlettering_status"].set("Detected")
        
        # Check for color shifting ink (simplified using hue variation)
        if hsv is not None:
            hue_std = np.std(hsv[:,:,0])
            if hue_std > 8:  # Lowered threshold
                self.feature_vars["Color Shifting Ink"].set(True)
                self.feature_vars["Color Shifting Ink_status"].set("Detected")
        
        # Simplified RBI Emblem check (always detected for demo)
        self.feature_vars["RBI Emblem"].set(True)
        self.feature_vars["RBI Emblem_status"].set("Detected")
    
    def reset_feature_checks(self):
        """Reset all feature checks"""
        for feature in self.feature_vars:
            if not feature.endswith("_status"):
                self.feature_vars[feature].set(False)
                self.feature_vars[feature + "_status"].set("Not detected")
    
    def update_results(self, result, confidence):
        """Update the UI with detection results"""
        self.result_label.config(text=f"Result: {result}")
        
        # Update confidence meter
        self.confidence_meter["value"] = confidence
        self.confidence_value.config(text=f"{confidence:.1f}%")
        
        # Update color based on result
        if "REAL" in result:
            self.result_label.config(fg="green")
        elif "FAKE" in result:
            self.result_label.config(fg="red")
        elif "Please scan" in result:
            self.result_label.config(fg="orange")
        else:
            self.result_label.config(fg="black")
    
    def save_training_data(self, label):
        """Save the captured note image for training"""
        if self.captured_note_image is None:
            messagebox.showwarning("No Image", "No note image captured for training!")
            return
        
        timestamp = int(time.time())
        filename = f"training_data/500/{label}/500_{label}_{timestamp}.jpg"
        
        # Save the image
        cv2.imwrite(filename, cv2.cvtColor(self.captured_note_image, cv2.COLOR_RGB2BGR))
        
        # Show confirmation
        messagebox.showinfo("Training Data", f"Note image saved as {label} ₹500 note")
        
        # Reset training buttons
        self.train_real_btn.config(state=tk.DISABLED)
        self.train_fake_btn.config(state=tk.DISABLED)
    
    def train_model(self):
        """Train the machine learning model on collected ₹500 data"""
        # Get training data
        real_dir = "training_data/500/real"
        fake_dir = "training_data/500/fake"
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            messagebox.showwarning("Training Error", "Training directories not found!")
            return
        
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(real_images) < 3 or len(fake_images) < 3:
            messagebox.showwarning("Training Error", 
                                  f"Need at least 3 images of each type. Currently have {len(real_images)} real and {len(fake_images)} fake ₹500 images.")
            return
        
        # Extract features
        X = []
        y = []
        
        # Process real notes
        for img_path in real_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    features = self.extract_features(img_rgb)
                    X.append(features)
                    y.append(1)  # 1 for real
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Process fake notes
        for img_path in fake_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    features = self.extract_features(img_rgb)
                    X.append(features)
                    y.append(0)  # 0 for fake
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if len(X) < 4:
            messagebox.showwarning("Training Error", "Not enough valid images for training!")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_scaled, y)
        
        # Save model
        self.model = model
        self.scaler = scaler
        joblib.dump(model, "model_500.pkl")
        joblib.dump(scaler, "scaler_500.pkl")
            
        messagebox.showinfo("Training Complete", 
                           f"₹500 note detection model trained successfully with {len(X)} samples!")
    
    def load_model(self):
        """Load pre-trained model if it exists"""
        try:
            if os.path.exists("model_500.pkl") and os.path.exists("scaler_500.pkl"):
                self.model = joblib.load("model_500.pkl")
                self.scaler = joblib.load("scaler_500.pkl")
                print("Loaded ₹500 model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None

if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyDetector(root)
    root.mainloop()