import cv2
import datetime
import subprocess
import webbrowser
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO  # Keeping your original YOLOv8
import pyttsx3
import speech_recognition as sr
from fuzzywuzzy import fuzz, process
import warnings
import threading
import time
import logging
import feedparser
import re
import random
import ast
import importlib.util
import queue
import sys
from gtts import gTTS
import os
from pygame import mixer
import math
from sympy import symbols, solve, sympify
from googletrans import Translator
import html
import pytesseract
import numpy as np
from deepface import DeepFace  # Still used for emotion detection
from transformers import pipeline  # Keeping your original pipeline
from PIL import Image, ImageDraw, ImageFont, ImageTk # For image generation
import face_recognition  # New: For advanced face recognition
from io import BytesIO  # New: For handling image downloads
import pickle  # New: For caching celebrity embeddings
import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame, Label, Button
import threading


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

# Check for required modules (all free and open-source)
required_modules = [
    "cv2", "ultralytics", "transformers", "pyttsx3", "speech_recognition", "fuzzywuzzy",
    "feedparser", "requests", "bs4", "gtts", "pygame", "sympy", "googletrans", "pytesseract",
    "numpy", "deepface", "PIL", "face_recognition"  # Added face_recognition
]
for module in required_modules:
    if not importlib.util.find_spec(module):
        logging.critical(f"Module '{module}' is missing. Install it with 'pip install {module}'.")
        raise ImportError(f"Missing required module: {module}")

class AuraAI:
    def __init__(self):
        """Initialize Aura AI with professional settings using free resources."""
        logging.info("Initializing Aura AI...")
        
        # Core components (all free)
        self.chatbot = None
        try:
            self.chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", torch_dtype="auto", device_map="auto")
            logging.info("Chatbot initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to load GPT-Neo: {e}. Falling back to basic responses.")
        
        self.object_detector = None
        try:
            self.object_detector = YOLO("yolov8n.pt")  # Your original YOLOv8 model
            logging.info("YOLO initialized successfully.")
        except Exception as e:
            logging.error(f"YOLO initialization failed: {e}. Object detection unavailable.")
        
        self.translator = None
        try:
            self.translator = Translator()  # Free Google Translate API via googletrans
            logging.info("Googletrans initialized successfully.")
        except Exception as e:
            logging.error(f"Googletrans initialization failed: {e}. Translation unavailable.")
        
        # Face recognition setup with face_recognition (new)
        self.celebrity_db = {}
        self.cache_file = "celebrity_embeddings.pkl"
        self._load_celebrity_database()  # Load or build the database
        
        # OCR setup (free Tesseract OCR)
        try:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path
            logging.info("Tesseract OCR initialized successfully.")
        except Exception as e:
            logging.error(f"Tesseract OCR initialization failed: {e}. Text reading unavailable.")
        
        # Health tips with a professional tone (your original)
        self.futuristic_tips = {
            "stress": [
                "Consider a holographic meditation. Visualize floating in a nebula to reduce stress effectively.",
                "A 5-minute zero-gravity nap, as suggested by nano-tech concepts, can refresh your mind."
            ],
            "energy": [
                "Imagine sipping quantum-infused water. It’s a creative way to boost your energy levels.",
                "Perform a set of star-jumps. This simple exercise can invigorate you quickly."
            ],
            "focus": [
                "Try a brainwave sync technique. Visualize a laser clearing mental distractions for better focus.",
                "Blink three times and refocus. It’s a proven method to sharpen your attention."
            ],
            "well-being": [
                "Practice slow, deep breathing like a Martian. It promotes a steady sense of calm.",
                "Take a moment to envision floating in a virtual galaxy. It’s an effective wellness reset."
            ],
            "sleep": [
                "Picture a lunar lullaby guiding you to rest. It’s a soothing way to improve sleep.",
                "A 10-minute orbit nap, inspired by nano-tech, can enhance your sleep quality."
            ],
            "anxiety": [
                "Visualize a calming photon wave enveloping you. It’s an effective anxiety reducer.",
                "Use a deep breathing technique—4 in, 4 hold, 8 out—for instant relief."
            ]
        }
        
        # Language and speech settings (your original)
        self.detected_lang = "en"
        self.recognition_lang = "en-US"
        self.speak_lang = "en"
        self.voice_pitch = 1.0
        
        self.last_response = ""
        self.tts_engine = None
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.use_gtts = False
        self.prefer_direct_speech = True
        self.awaiting_creator_followup = False
        
        mixer.init()
        
        try:
            self.tts_engine = pyttsx3.init('sapi5')  # Free SAPI5 TTS engine
            self.tts_engine.setProperty("rate", 170)
            zira_voice = self.get_zira_voice()
            if zira_voice:
                self.tts_engine.setProperty("voice", zira_voice)
                logging.info("Set voice to Microsoft Zira Desktop.")
            else:
                logging.warning("Microsoft Zira Desktop not found. Falling back to default voice.")
                self.tts_engine.setProperty("voice", self.get_female_voice())
            self._start_speech_thread()
            logging.info("Text-to-speech initialized successfully.")
            self.speak("Greetings. This is Aura, testing audio functionality with Microsoft Zira Desktop. Can you hear me?")
            logging.info("Initial TTS test completed.")
        except Exception as e:
            logging.error(f"TTS setup failed: {e}. Switching to gTTS.")
            self.use_gtts = True
        
        self.microphone = None
        self.recognizer = None
        try:
            self.microphone = sr.Microphone()
            self.recognizer = sr.Recognizer()  # Free Google Speech Recognition API
            logging.info("Microphone and recognizer initialized successfully.")
        except Exception as e:
            logging.error(f"Microphone setup failed: {e}. Voice input disabled.")
        
        self.sentiment_analyzer = None
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            logging.info("Sentiment analyzer initialized successfully.")
        except Exception as e:
            logging.error(f"Sentiment analyzer failed: {e}. Mood tracking disabled.")
        
        self.memory = []
        self.user_mood = "neutral"
        self.reminders = []
        self.custom_greeting = None
        self.current_game = None
        self.guessing_number = None
        self.trivia_question = None
        self.trivia_correct_answer = None
        self.trivia_options = []
        self.current_emotion = "neutral"  # Track real-time emotion from face


        # GUI Setup
        self.root = tk.Tk()
        self.root.title("Aura AI Interface")
        self.root.geometry("1000x700")  # Larger window for more elements
        self.root.configure(bg="#1a1a1a")  # Darker, sleek background
        
        # Main Frame Layout
        self.main_frame = Frame(self.root, bg="#1a1a1a")
        self.main_frame.pack(fill="both", expand=True)

        # Left Panel: Command History
        self.history_frame = Frame(self.main_frame, bg="#2a2a2a", width=300)
        self.history_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.history_label = Label(self.history_frame, text="Command History", fg="cyan", bg="#2a2a2a", font=("Arial", 14, "bold"))
        self.history_label.pack(pady=5)
        self.history_canvas = Canvas(self.history_frame, bg="#2a2a2a", highlightthickness=0)
        self.history_scrollbar = Scrollbar(self.history_frame, orient="vertical", command=self.history_canvas.yview)
        self.history_inner_frame = Frame(self.history_canvas, bg="#2a2a2a")
        self.history_canvas.configure(yscrollcommand=self.history_scrollbar.set)
        self.history_canvas.pack(side="left", fill="both", expand=True)
        self.history_scrollbar.pack(side="right", fill="y")
        self.history_canvas.create_window((0, 0), window=self.history_inner_frame, anchor="nw")
        self.history_items = []

        # Right Panel: Main Display
        self.display_frame = Frame(self.main_frame, bg="#1a1a1a")
        self.display_frame.pack(side="right", fill="both", expand=True)

        # Central Sphere
        self.canvas = Canvas(self.display_frame, width=600, height=400, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(pady=20)
        self.sphere_radius = 60
        self.sphere_x = 300
        self.sphere_y = 200
        self.sphere = None
        self.active = False
        self.pulse_size = 0
        self.pulse_direction = 1

        # Status Bar
        self.status_frame = Frame(self.display_frame, bg="#2a2a2a")
        self.status_frame.pack(fill="x", pady=10)
        self.status_label = Label(self.status_frame, text="Status: Idle", fg="white", bg="#2a2a2a", font=("Arial", 12))
        self.status_label.pack(side="left", padx=10)
        self.mic_button = Button(self.status_frame, text="Mic: ON", fg="white", bg="#4a4a4a", command=self.toggle_mic, font=("Arial", 10))
        self.mic_button.pack(side="right", padx=10)

        # Popup Storage
        self.popup_screens = []

        # Start GUI
        self.greet_user()
        self.announce_date_and_day()
        self.draw_sphere()
        self.animate_sphere()
        self.canvas.bind("<Button-1>", self.activate_ai)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def draw_sphere(self):
        """Draw a glowing, futuristic sphere."""
        if self.sphere:
            self.canvas.delete(self.sphere)
        gradient_radius = self.sphere_radius + self.pulse_size + 20
        self.canvas.create_oval(
            self.sphere_x - gradient_radius, self.sphere_y - gradient_radius,
            self.sphere_x + gradient_radius, self.sphere_y + gradient_radius,
            fill="#003366", outline=""  # Subtle glow effect
        )
        self.sphere = self.canvas.create_oval(
            self.sphere_x - self.sphere_radius - self.pulse_size,
            self.sphere_y - self.sphere_radius - self.pulse_size,
            self.sphere_x + self.sphere_radius + self.pulse_size,
            self.sphere_y + self.sphere_radius + self.pulse_size,
            fill="cyan" if self.active else "#00b7eb",
            outline="white", width=2
        )
        self.canvas.create_text(self.sphere_x, self.sphere_y, text="AURA", fill="white", font=("Arial", 20, "bold"))

    def animate_sphere(self):
        """Animate the sphere with a pulsing glow."""
        self.pulse_size += self.pulse_direction * 2
        if self.pulse_size > 15 or self.pulse_size < 0:
            self.pulse_direction *= -1
        self.draw_sphere()
        self.root.after(50, self.animate_sphere)

    def activate_ai(self, event):
        """Activate AI on sphere click."""
        if not self.active and abs(event.x - self.sphere_x) < self.sphere_radius and abs(event.y - self.sphere_y) < self.sphere_radius:
            self.active = True
            self.status_label.config(text="Status: Listening")
            self.draw_sphere()
            self.speak("Aura AI activated. Please provide a command.")
            threading.Thread(target=self.listen_for_commands, daemon=True).start()

    def toggle_mic(self):
        """Toggle microphone state."""
        self.active = not self.active
        self.mic_button.config(text=f"Mic: {'ON' if self.active else 'OFF'}")
        self.status_label.config(text=f"Status: {'Listening' if self.active else 'Idle'}")
        if self.active:
            self.speak("Microphone activated.")
            threading.Thread(target=self.listen_for_commands, daemon=True).start()
        else:
            self.speak("Microphone deactivated.")

    def listen_for_commands(self):
        """Listen for commands and update GUI."""
        while self.active:
            command = self.listen_to_voice()
            if command in ["exit", "quit"]:
                self.active = False
                self.status_label.config(text="Status: Idle")
                self.speak("Deactivating Aura AI.")
                self.clear_popups()
                self.draw_sphere()
                break
            elif command not in ["silence", "unintelligible", "mic_error", "speech_service_down", "audio_error"]:
                self.status_label.config(text=f"Status: Processing '{command}'")
                response = self.process_command(command, command)
                self.add_to_history(command, response)
                self.show_popup(command, response)
            else:
                self.status_label.config(text="Status: Listening")

    def add_to_history(self, command, response):
        """Add command and response to history panel."""
        item_frame = Frame(self.history_inner_frame, bg="#2a2a2a", pady=5)
        item_frame.pack(fill="x", padx=5)
        cmd_label = Label(item_frame, text=f"> {command}", fg="cyan", bg="#2a2a2a", font=("Arial", 10), wraplength=250, justify="left")
        cmd_label.pack(anchor="w")
        resp_label = Label(item_frame, text=f"{response[:50]}...", fg="white", bg="#2a2a2a", font=("Arial", 10), wraplength=250, justify="left")
        resp_label.pack(anchor="w")
        self.history_items.append(item_frame)
        self.history_canvas.update_idletasks()
        self.history_canvas.config(scrollregion=self.history_canvas.bbox("all"))
        self.history_canvas.yview_moveto(1.0)

    def show_popup(self, command, response):
        """Display command results in a styled popup."""
        self.clear_popups()
        popup_frame = Frame(self.display_frame, bg="#2d2d2d", bd=2, relief="raised")
        popup_frame.place(x=200, y=450, width=400, height=200)  # Fixed position below sphere
        Label(popup_frame, text=f"Command: {command}", fg="cyan", bg="#2d2d2d", font=("Arial", 12, "bold")).pack(pady=5)
        Label(popup_frame, text=f"Result: {response[:100]}...", fg="white", bg="#2d2d2d", font=("Arial", 10), wraplength=380).pack(pady=5)
        self.popup_screens.append(popup_frame)

        # Specialized Displays
        if "recognize face" in command.lower():
            self.show_face_recognition(popup_frame)
        elif "detect objects" in command.lower():
            self.show_object_detection(popup_frame)
        elif "generate image" in command.lower():
            self.show_generated_image(popup_frame)

    def clear_popups(self):
        """Remove all popups."""
        for popup in self.popup_screens:
            popup.destroy()
        self.popup_screens = []

    def show_face_recognition(self, popup_frame):
        """Display face recognition results."""
        # ... (keep your existing face recognition logic) ...
        result = "Recognized: Sample Name"  # Replace with actual result
        Label(popup_frame, text=result, fg="yellow", bg="#2d2d2d", font=("Arial", 10)).pack(pady=5)

    def show_object_detection(self, popup_frame):
        """Display object detection results."""
        # ... (keep your existing object detection logic) ...
        result = "Detected: Person, Car"  # Replace with actual result
        Label(popup_frame, text=result, fg="yellow", bg="#2d2d2d", font=("Arial", 10)).pack(pady=5)

    def show_generated_image(self, popup_frame):
        """Display generated image."""
        img = self._generate_text_image("Sample Image")
        img.save("temp_image.png")
        photo = ImageTk.PhotoImage(file="temp_image.png")
        img_label = Label(popup_frame, image=photo, bg="#2d2d2d")
        img_label.image = photo  # Keep reference
        img_label.pack(pady=5)
        self.popup_screens.append(img_label)

    def on_closing(self):
        """Handle window close."""
        self.active = False
        self.speak("Shutting down Aura AI.")
        self.root.destroy()

    # ... (rest of your existing methods remain unchanged) ...

    def draw_sphere(self):
        """Draw the animated sphere in the center."""
        if self.sphere:
            self.canvas.delete(self.sphere)
        self.sphere = self.canvas.create_oval(
            self.sphere_x - self.sphere_radius - self.pulse_size,
            self.sphere_y - self.sphere_radius - self.pulse_size,
            self.sphere_x + self.sphere_radius + self.pulse_size,
            self.sphere_y + self.sphere_radius + self.pulse_size,
            fill="cyan" if self.active else "blue",
            outline="white",
            width=2
        )

    def animate_sphere(self):
        """Animate the sphere with a pulsing effect."""
        self.pulse_size += self.pulse_direction * 2
        if self.pulse_size > 10 or self.pulse_size < 0:
            self.pulse_direction *= -1
        self.draw_sphere()
        self.root.after(50, self.animate_sphere)

    def activate_ai(self, event):
        """Activate the AI when the sphere is clicked."""
        if not self.active and abs(event.x - self.sphere_x) < self.sphere_radius and abs(event.y - self.sphere_y) < self.sphere_radius:
            self.active = True
            self.draw_sphere()
            self.speak("Aura AI activated. Please provide a command.")
            threading.Thread(target=self.listen_for_commands, daemon=True).start()

    def listen_for_commands(self):
        """Listen for voice commands and display results in pop-up screens."""
        while self.active:
            command = self.listen_to_voice()
            if command in ["exit", "quit"]:
                self.active = False
                self.speak("Deactivating Aura AI.")
                self.clear_popups()
                self.draw_sphere()
                break
            elif command not in ["silence", "unintelligible", "mic_error", "speech_service_down", "audio_error"]:
                response = self.process_command(command, command)
                self.show_popup(command, response)

    def show_popup(self, command, response):
        """Display command results in an animated pop-up screen."""
        self.clear_popups()
        angle = random.uniform(0, 2 * 3.14159)
        distance = 100
        popup_x = self.sphere_x + distance * np.cos(angle)
        popup_y = self.sphere_y + distance * np.sin(angle)
        
        popup = self.canvas.create_rectangle(
            popup_x - 100, popup_y - 50, popup_x + 100, popup_y + 50,
            fill="darkblue", outline="cyan", width=2
        )
        self.popup_screens.append(popup)
        
        cmd_text = self.canvas.create_text(popup_x, popup_y - 20, text=f"Command: {command}", fill="white", font=("Arial", 10))
        resp_text = self.canvas.create_text(popup_x, popup_y + 20, text=f"Result: {response[:50]}...", fill="white", font=("Arial", 10))
        self.popup_screens.extend([cmd_text, resp_text])
        
        if "recognize face" in command.lower():
            self.show_face_recognition(popup_x, popup_y)
        elif "detect objects" in command.lower():
            self.show_object_detection(popup_x, popup_y)
        elif "generate image" in command.lower():
            self.show_generated_image(popup_x, popup_y)

    def clear_popups(self):
        """Remove all pop-up screens."""
        for item in self.popup_screens:
            self.canvas.delete(item)
        self.popup_screens = []

    def show_face_recognition(self, x, y):
        """Display face recognition results in the popup."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.speak("Camera unavailable.")
            return
        
        recognized_names = set()
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                for encoding in face_encodings:
                    for name, celeb_embedding in self.celebrity_db.items():
                        if face_recognition.compare_faces([celeb_embedding], encoding, tolerance=0.6)[0]:
                            recognized_names.add(name)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        result = f"Recognized: {', '.join(recognized_names)}" if recognized_names else "No faces recognized."
        self.canvas.create_text(x, y + 40, text=result, fill="yellow", font=("Arial", 10))

    def show_object_detection(self, x, y):
        """Display object detection results in the popup."""
        if not self.object_detector:
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.speak("Camera unavailable.")
            return
        
        objects = set()
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.object_detector(frame, conf=0.3)
            for result in results:
                for box in result.boxes:
                    label = self.object_detector.names[int(box.cls[0])]
                    objects.add(label)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        result = f"Detected: {', '.join(objects)}" if objects else "No objects detected."
        self.canvas.create_text(x, y + 40, text=result, fill="yellow", font=("Arial", 10))

    def show_generated_image(self, x, y):
        """Display a generated image in the popup."""
        img = self._generate_text_image("Sample Image")
        img.save("temp_image.png")
        photo = ImageTk.PhotoImage(file="temp_image.png")
        img_item = self.canvas.create_image(x, y + 60, image=photo)
        self.popup_screens.append(img_item)
        self.root.image_references = getattr(self.root, 'image_references', [])
        self.root.image_references.append(photo)  # Prevent garbage collection    

    def _fetch_image_from_url(self, url):
        """Helper method to download images for celebrity database."""
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return np.array(img)
        except Exception as e:
            logging.error(f"Failed to fetch image from {url}: {e}")
            return None

    def _load_celebrity_database(self):
        """Load all 68 celebrities into the database, looping until every single one is successfully loaded."""
        total_celebs = 68  # Exact number of celebrities to load
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        max_attempts_per_celeb = 10  # Maximum retries per celebrity per loop iteration

        # Load from cache if it exists and is complete
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.celebrity_db = pickle.load(f)
            logging.info(f"Loaded {len(self.celebrity_db)} celebrities from cache.")
            if len(self.celebrity_db) == total_celebs:
                logging.info("Cache is complete with all 68 celebrities.")
                return
            else:
                logging.warning(f"Cache incomplete ({len(self.celebrity_db)}/{total_celebs}), rebuilding from scratch...")
                self.celebrity_db = {}  # Start fresh if cache is incomplete

        # Full list of celebrities with URLs
        celebrities = {
            # Indian Politicians
            "Narendra Modi": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Narendra_Modi_2023.jpg/220px-Narendra_Modi_2023.jpg",
            "Nitin Gadkari": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Nitin_Gadkari_in_2023.jpg/220px-Nitin_Gadkari_in_2023.jpg",
            "Amit Shah": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Amit_Shah_in_2023.jpg/220px-Amit_Shah_in_2023.jpg",
            "Rahul Gandhi": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Rahul_Gandhi_in_2018.jpg/220px-Rahul_Gandhi_in_2018.jpg",
            "Mamata Banerjee": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Mamata_Banerjee_-_Kolkata_2011-12-09_5174.JPG/220px-Mamata_Banerjee_-_Kolkata_2011-12-09_5174.JPG",
            "Arvind Kejriwal": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Arvind_Kejriwal_2023.jpg/220px-Arvind_Kejriwal_2023.jpg",
            "Yogi Adityanath": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Yogi_Adityanath_in_2023.jpg/220px-Yogi_Adityanath_in_2023.jpg",
            "Sonia Gandhi": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Sonia_Gandhi_in_2014.jpg/220px-Sonia_Gandhi_in_2014.jpg",
            "Pawan Kalyan": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Pawan_Kalyan_in_2023.jpg/220px-Pawan_Kalyan_in_2023.jpg",
            "Smriti Irani": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Smriti_Irani_in_2023.jpg/220px-Smriti_Irani_in_2023.jpg",

            # Indian Bollywood Actors
            "Shah Rukh Khan": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Shah_Rukh_Khan_graces_the_launch_event_of_the_new_Santro.jpg/220px-Shah_Rukh_Khan_graces_the_launch_event_of_the_new_Santro.jpg",
            "Deepika Padukone": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Deepika_Padukone_at_an_event_in_2018.jpg/220px-Deepika_Padukone_at_an_event_in_2018.jpg",
            "Aamir Khan": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Aamir_Khan_in_2017.jpg/220px-Aamir_Khan_in_2017.jpg",
            "Salman Khan": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Salman_Khan_at_Renault_Star_Guild_Awards.jpg/220px-Salman_Khan_at_Renault_Star_Guild_Awards.jpg",
            "Priyanka Chopra": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Priyanka_Chopra_in_2019.jpg/220px-Priyanka_Chopra_in_2019.jpg",
            "Amitabh Bachchan": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Amitabh_Bachchan_2018.jpg/220px-Amitabh_Bachchan_2018.jpg",
            "Ranbir Kapoor": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ranbir_Kapur_promoting_Brahmastra.jpg/220px-Ranbir_Kapur_promoting_Brahmastra.jpg",
            "Alia Bhatt": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Alia_Bhatt_at_an_event_for_Razi_in_2018_%28cropped%29.jpg/220px-Alia_Bhatt_at_an_event_for_Razi_in_2018_%28cropped%29.jpg",
            "Akshay Kumar": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Akshay_Kumar_in_2020.jpg/220px-Akshay_Kumar_in_2020.jpg",
            "Hrithik Roshan": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Hrithik_Roshan_graces_Vogue_Beauty_Awards.jpg/220px-Hrithik_Roshan_graces_Vogue_Beauty_Awards.jpg",
            "Katrina Kaif": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Katrina_Kaif_in_2018.jpg/220px-Katrina_Kaif_in_2018.jpg",
            "Anushka Sharma": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Anushka_Sharma_in_2018.jpg/220px-Anushka_Sharma_in_2018.jpg",
            "Ajay Devgn": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Ajay_Devgn_promoting_Tanhaji.jpg/220px-Ajay_Devgn_promoting_Tanhaji.jpg",
            "Ranveer Singh": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Ranveer_Singh_in_2023.jpg/220px-Ranveer_Singh_in_2023.jpg",
            "Kareena Kapoor": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Kareena_Kapoor_in_2020.jpg/220px-Kareena_Kapoor_in_2020.jpg",
            "Aishwarya Rai": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Aishwarya_Rai_in_2016.jpg/220px-Aishwarya_Rai_in_2016.jpg",

            # Indian South Cinema Actors
            "Rajinikanth": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Rajinikanth_2018.jpg/220px-Rajinikanth_2018.jpg",
            "Kamal Haasan": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Kamal_Haasan_in_2017.jpg/220px-Kamal_Haasan_in_2017.jpg",
            "Vijay": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Vijay_at_the_Leo_Success_Meet.jpg/220px-Vijay_at_the_Leo_Success_Meet.jpg",
            "Prabhas": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Prabhas_2019_%28cropped%29.jpg/220px-Prabhas_2019_%28cropped%29.jpg",
            "Allu Arjun": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Allu_Arjun_at_pushpa_interview.jpg/220px-Allu_Arjun_at_pushpa_interview.jpg",
            "Dhanush": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Dhanush_at_the_audio_launch_of_Velaiilla_Pattadhari.jpg/220px-Dhanush_at_the_audio_launch_of_Velaiilla_Pattadhari.jpg",
            "Mahesh Babu": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Mahesh_Babu_in_2019.jpg/220px-Mahesh_Babu_in_2019.jpg",
            "Tamannaah Bhatia": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Tamannaah_Bhatia_in_2023.jpg/220px-Tamannaah_Bhatia_in_2023.jpg",
            "Mohanlal": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Mohanlal_in_2018.jpg/220px-Mohanlal_in_2018.jpg",
            "Mammootty": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Mammootty_in_2016.jpg/220px-Mammootty_in_2016.jpg",
            "NTR Jr": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/N._T._Rama_Rao_Jr._in_2023.jpg/220px-N._T._Rama_Rao_Jr._in_2023.jpg",
            "Ram Charan": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Ram_Charan_in_2023.jpg/220px-Ram_Charan_in_2023.jpg",
            "Rashmika Mandanna": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Rashmika_Mandanna_in_2023.jpg/220px-Rashmika_Mandanna_in_2023.jpg",

            # Indian Cricketers
            "Virat Kohli": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Virat_Kohli_in_2023.jpg/220px-Virat_Kohli_in_2023.jpg",
            "Sachin Tendulkar": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Sachin_Tendulkar_in_2019.jpg/220px-Sachin_Tendulkar_in_2019.jpg",
            "MS Dhoni": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/MS_Dhoni_in_2016.jpg/220px-MS_Dhoni_in_2016.jpg",
            "Kapil Dev": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Kapil_Dev_in_2023.jpg/220px-Kapil_Dev_in_2023.jpg",
            "Rohit Sharma": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Rohit_Sharma_in_2023.jpg/220px-Rohit_Sharma_in_2023.jpg",
            "Jasprit Bumrah": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Jasprit_Bumrah_in_2023.jpg/220px-Jasprit_Bumrah_in_2023.jpg",
            "Smriti Mandhana": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Smriti_Mandhana_in_2018.jpg/220px-Smriti_Mandhana_in_2018.jpg",
            "Yuvraj Singh": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Yuvraj_Singh_in_2017.jpg/220px-Yuvraj_Singh_in_2017.jpg",

            # Indian Musicians and Others
            "Sonu Nigam": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sonu_Nigam_2017.jpg/220px-Sonu_Nigam_2017.jpg",
            "Shreya Ghoshal": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Shreya_Ghoshal_in_2018.jpg/220px-Shreya_Ghoshal_in_2018.jpg",
            "AR Rahman": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/A._R._Rahman_in_2019.jpg/220px-A._R._Rahman_in_2019.jpg",
            "Sania Mirza": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Sania_Mirza_in_2017.jpg/220px-Sania_Mirza_in_2017.jpg",

            # Global Celebrities
            "Elon Musk": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_2023.jpg/220px-Elon_Musk_2023.jpg",
            "Beyoncé": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Beyonce_-_The_Formation_World_Tour%2C_at_Wembley_Stadium_in_London%2C_England_%2829275153062%29_%28cropped%29.jpg/220px-Beyonce_-_The_Formation_World_Tour%2C_at_Wembley_Stadium_in_London%2C_England_%2829275153062%29_%28cropped%29.jpg",
            "Cristiano Ronaldo": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Cristiano_Ronaldo_2018.jpg/220px-Cristiano_Ronaldo_2018.jpg",
            "Tom Hanks": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Tom_Hanks_2016.jpg/220px-Tom_Hanks_2016.jpg",
            "Serena Williams": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Serena_Williams_2018.jpg/220px-Serena_Williams_2018.jpg",
            "Barack Obama": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/220px-President_Barack_Obama.jpg",
            "Lionel Messi": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg/220px-Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg",
            "Angelina Jolie": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Angelina_Jolie_2_June_2014_%28cropped%29.jpg/220px-Angelina_Jolie_2_June_2014_%28cropped%29.jpg",
            "Justin Bieber": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Justin_Bieber_in_2015.jpg/220px-Justin_Bieber_in_2015.jpg",
            "Taylor Swift": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.jpg/220px-Taylor_Swift_at_the_2019_American_Music_Awards_%28cropped%29.jpg",
            "Brad Pitt": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg/220px-Brad_Pitt_2019_by_Glenn_Francis.jpg",
            "Scarlett Johansson": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Scarlett_Johansson_in_2019.jpg/220px-Scarlett_Johansson_in_2019.jpg",
            "Leonardo DiCaprio": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Leonardo_DiCaprio_2016.jpg/220px-Leonardo_DiCaprio_2016.jpg",
            "Oprah Winfrey": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Oprah_Winfrey_in_2014.jpg/220px-Oprah_Winfrey_in_2014.jpg",
            "Joe Biden": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Joe_Biden_2021.jpg/220px-Joe_Biden_2021.jpg",
            "Emma Watson": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Emma_Watson_2017.jpg/220px-Emma_Watson_2017.jpg",
            "Will Smith": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Will_Smith_2019.jpg/220px-Will_Smith_2019.jpg",
            "Rihanna": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Rihanna_in_2018.jpg/220px-Rihanna_in_2018.jpg",
        }
        base_url = "https://en.wikipedia.org/wiki/"
        headers = {"User-Agent": "Mozilla/5.0"}

        for celeb in celebrities:
            try:
                url = base_url + celeb.replace(" ", "_")
                response = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")
                img_tag = soup.find("img", class_="mw-file-element")
                if img_tag and "src" in img_tag.attrs:
                    img_url = "https:" + img_tag["src"]
                    img_array = self._fetch_image_from_url(img_url)
                    if img_array is None:
                        continue
                    face_encodings = face_recognition.face_encodings(img_array)
                    if face_encodings:
                        self.celebrity_db[celeb] = face_encodings[0]
            except Exception as e:
                logging.error(f"Error loading {celeb}: {e}")

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.celebrity_db, f)
        logging.info(f"Loaded and cached {len(self.celebrity_db)} celebrities.")

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

    def _start_speech_thread(self):
        """Start the speech thread for queued responses."""
        if self.tts_engine and (self.speech_thread is None or not self.speech_thread.is_alive()):
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
            logging.info("Speech thread started or restarted.")

    def _speech_worker(self):
        """Worker thread to process speech queue."""
        while True:
            try:
                text = self.speech_queue.get(timeout=10)
                if text is None:
                    logging.info("Speech thread received None, stopping.")
                    break
                if self.tts_engine and not self.use_gtts and self.speak_lang == "en":
                    logging.info(f"Speaking from queue: {text[:50]}...")
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.last_response = text
                    logging.info(f"Successfully spoke from queue: {text[:50]}...")
                else:
                    self._speak_gtts(text)
                self.speech_queue.task_done()
            except queue.Empty:
                logging.warning("Speech queue empty for too long, thread may be stalled.")
                break
            except Exception as e:
                logging.error(f"TTS error in worker: {e}")
                print(f"Text fallback due to worker error: {text}")

    def get_zira_voice(self):
        """Select Microsoft Zira Desktop - English (United States) voice."""
        if not self.tts_engine:
            return None
        try:
            voices = self.tts_engine.getProperty("voices")
            for voice in voices:
                if voice.name == "Microsoft Zira Desktop - English (United States)":
                    logging.info(f"Found Microsoft Zira Desktop voice: {voice.name}")
                    return voice.id
            logging.warning("Microsoft Zira Desktop voice not found among available voices.")
            return None
        except Exception as e:
            logging.error(f"Error finding Zira voice: {e}")
            return None

    def get_female_voice(self):
        """Fallback method to select a professional female voice if Zira is unavailable."""
        if not self.tts_engine:
            return None
        try:
            voices = self.tts_engine.getProperty("voices")
            for voice in voices:
                if "female" in voice.name.lower():
                    logging.info(f"Selected fallback female voice: {voice.name}")
                    return voice.id
            logging.info(f"No female voice found, using default: {voices[0].name}")
            return voices[0].id
        except Exception:
            logging.warning("Voice selection failed.")
            return None

    def greet_user(self):
        """Greet the user with a professional tone."""
        if self.custom_greeting:
            self.speak(self.custom_greeting)
        else:
            greetings = [
                "Greetings. I’m Aura, your virtual assistant. How may I assist you today?",
                "Hello. I’m Aura, here to provide support. What can I do for you?",
                "Good day. I’m Aura, your automated assistant. How can I help you?"
            ]
            self.speak(random.choice(greetings))

    def announce_date_and_day(self):
        """Announce the current date and day after initialization."""
        try:
            now = datetime.datetime.now()
            date_str = now.strftime("%B %d, %Y")
            day_str = now.strftime("%A")
            self.speak(f"Today is {day_str}, {date_str}.")
        except Exception as e:
            logging.error(f"Date and day announcement error: {e}")
            self.speak("Unable to retrieve the current date and day.")

    def introduce_self(self, command):
        """Introduce Aura, its creator, and its capabilities."""
        intro = (
            "I am Aura, an advanced virtual assistant created by Mukka Srivatsav, a second-year BTech student studying "
            "Artificial Intelligence and Machine Learning at Sphoorthy Engineering College, born on November 30, 2005. "
            "I’m designed to assist with tasks including face recognition, text reading from objects, answering questions, "
            "translating languages, generating code, solving math problems, fetching news and weather updates, telling jokes, "
            "setting reminders, and playing games like guessing and trivia. I can also detect objects in your surroundings, "
            "provide health tips, define words, and open applications. How may I assist you today?"
        )
        return intro

    def why_name_aura(self, command):
        """Answer why the name 'Aura' was chosen."""
        response = (
            "My name is Aura because my creator, Mukka Sri Vatsav, chose it for me. He was inspired by the name Jarvis from Iron Man, "
            "but wanted something unique. In mythological terms, 'Aura' means positive energy, which reflects my purpose. It also stands "
            "for Augmented Utility Reality Assistant, highlighting my role as a helpful AI companion. Do you want to know more about my creator?"
        )
        self.awaiting_creator_followup = True
        return response

    def creator_details(self, command):
        """Provide detailed information about the creator if the user says 'yes'."""
        if "yes" in command.lower() and self.awaiting_creator_followup:
            self.awaiting_creator_followup = False
            details = (
                "My creator, Sri Vatsav, is very fond of AI. His first project was an OS Help Chatbot using Telegram as an interface—"
                "a brilliant idea, isn’t it? He built it without using any paid resources, and it could solve any hardware or software issue "
                "related to operating systems. Quite impressive, right? How else can I assist you now?"
            )
            return details
        elif "no" in command.lower() and self.awaiting_creator_followup:
            self.awaiting_creator_followup = False
            return "Alright, I won’t go into more details about my creator. How can I assist you further?"
        else:
            self.awaiting_creator_followup = False
            return None

    def speak(self, text, force_lang=None):
        """Speak with a professional tone and proper punctuation, with optional language override."""
        if not isinstance(text, str) or not text:
            text = "No input provided. How may I assist you?"
        if text == self.last_response:
            text = "I’ve already addressed that. How else can I assist you?"
        if not text.endswith(('.', '!', '?')):
            text += "."
        
        speak_lang = force_lang if force_lang else self.speak_lang
        logging.info(f"Preparing to speak in {speak_lang}: {text[:50]}...")
        
        if self.tts_engine and not self.use_gtts and speak_lang == "en":
            self.tts_engine.setProperty("rate", int(170 * self.voice_pitch))
            for attempt in range(3):
                try:
                    logging.info(f"Attempting direct speech (attempt {attempt + 1}): {text[:50]}...")
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.last_response = text
                    logging.info(f"Direct speech successful: {text[:50]}...")
                    return
                except Exception as e:
                    logging.error(f"Direct speech failed (attempt {attempt + 1}): {e}")
                    if attempt < 2:
                        try:
                            self.tts_engine = pyttsx3.init('sapi5')
                            self.tts_engine.setProperty("rate", int(170 * self.voice_pitch))
                            zira_voice = self.get_zira_voice()
                            self.tts_engine.setProperty("voice", zira_voice if zira_voice else self.get_female_voice())
                            logging.info("TTS engine reinitialized for retry.")
                        except Exception as reinit_e:
                            logging.error(f"Reinitialization failed: {reinit_e}")
                            break
        
        for attempt in range(2):
            try:
                self._speak_gtts(text, attempt, lang=speak_lang)
                return
            except Exception as e:
                logging.error(f"gTTS attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    logging.error("All gTTS attempts failed, falling back to text.")
                    print(f"Text fallback: {text}")
                    self.last_response = text

    def _speak_gtts(self, text, attempt=0, lang="en"):
        """Speak using gTTS with a professional delivery."""
        temp_file = f"temp_audio_{attempt}_{int(time.time())}.mp3"
        try:
            tts = gTTS(text=text, lang=lang if lang in ['en', 'hi', 'te', 'ta', 'ml', 'kn', 'es'] else 'en', slow=False)
            tts.save(temp_file)
            mixer.music.load(temp_file)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            mixer.music.unload()
            os.remove(temp_file)
            logging.info(f"Spoke via gTTS in {lang}: {text[:50]}...")
            self.last_response = text
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise e

    def test_speech(self, command=None):
        """Test the speech system with a professional message."""
        test_text = "This is Aura, conducting an audio test. Can you hear me clearly?"
        logging.info("Running speech test...")
        try:
            if not self.use_gtts and self.speak_lang == "en":
                self.tts_engine.say(test_text)
                self.tts_engine.runAndWait()
                logging.info("pyttsx3 test successful.")
                return "Speech test completed. Was the audio clear?"
            else:
                self._speak_gtts(test_text)
                return "Google TTS test completed. Was the audio audible?"
        except Exception as e:
            logging.error(f"Speech test failed: {e}")
            self.use_gtts = True
            return "Speech test failed. Switching to Google TTS. Please retry with 'test speech'."

    def toggle_tts_engine(self, command):
        """Toggle between TTS engines with a professional response."""
        if "use pyttsx3" in command.lower():
            self.use_gtts = False
            self.prefer_direct_speech = True
            self.speak("Switched to pyttsx3 engine. Testing audio now.")
        elif "use gtts" in command.lower():
            self.use_gtts = True
            self.prefer_direct_speech = False
            self.speak("Switched to Google TTS engine. Testing audio now.")
        else:
            self.speak("Please specify 'use pyttsx3' or 'use gtts' to switch engines.")
        return "TTS engine toggled successfully."

    def set_voice_pitch(self, command):
        """Adjust voice pitch with a professional response."""
        try:
            pitch_match = re.search(r"set voice pitch to (\d+\.?\d*)", command.lower())
            if pitch_match:
                new_pitch = float(pitch_match.group(1))
                if 0.5 <= new_pitch <= 2.0:
                    self.voice_pitch = new_pitch
                    self.speak(f"Voice pitch set to {new_pitch}. Does this meet your preference?")
                else:
                    self.speak("Pitch must be between 0.5 and 2.0. Please try again.")
            else:
                self.speak("Please provide a command like 'set voice pitch to 1.5'.")
        except Exception as e:
            logging.error(f"Voice pitch error: {e}")
            return "Unable to adjust pitch. Please retry."

    def set_language(self, command):
        """Set the language with a professional tone."""
        lang_map = {
            "english": ("en-US", "en"),
            "hindi": ("hi-IN", "hi"),
            "telugu": ("te-IN", "te"),
            "tamil": ("ta-IN", "ta"),
            "malayalam": ("ml-IN", "ml"),
            "kannada": ("kn-IN", "kn"),
            "spanish": ("es-ES", "es")
        }
        match = re.search(r"(set|change)\s+language\s+to\s+(.+)", command.lower())
        if match:
            lang_name = match.group(2).strip().lower()
            if lang_name in lang_map:
                self.recognition_lang, self.speak_lang = lang_map[lang_name]
                if lang_name == "english":
                    self.speak("Language set to English. How may I assist you now?")
                elif lang_name == "hindi":
                    self.speak("भाषा हिंदी में सेट की गई है। मैं आपकी अब कैसे सहायता कर सकती हूँ?")
                elif lang_name == "telugu":
                    self.speak("భాష తెలుగులో సెట్ చేయబడింది. నేను ఇప్పుడు మీకు ఎలా సహాయపడగలను?")
                elif lang_name == "tamil":
                    self.speak("மொழி தமிழுக்கு அமைக்கப்பட்டது. இப்போது உங்களுக்கு எவ்வாறு உதவலாம்?")
                elif lang_name == "malayalam":
                    self.speak("ഭാഷ മലയാളത്തിലേക്ക് സെറ്റ് ചെയ്തു. ഞാൻ ഇപ്പോൾ നിന്നെ എങ്ങനെ സഹായിക്കാം?")
                elif lang_name == "kannada":
                    self.speak("ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಸೆಟ್ ಮಾಡಲಾಗಿದೆ. ನಾನು ಈಗ ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?")
                elif lang_name == "spanish":
                    self.speak("El idioma ha sido configurado a español. ¿En qué puedo ayudarte ahora?")
                return None
            else:
                self.speak(f"Language '{lang_name}' is not supported. Available options are English, Hindi, Telugu, Tamil, Malayalam, Kannada, and Spanish.")
        return "Please say 'set language to [language]' to change the language."

    def generate_code(self, command):
        """Generate code with a professional response."""
        try:
            prompt = re.sub(r"^(generate|write|show\s+me)( a)?\s+code\s*(for|to|on)?\s*", "", command, flags=re.IGNORECASE).strip()
            if not prompt:
                return "Please specify what code to generate, such as 'generate code to sort a list'."
        
            fallback_code = {
                "factorial": """def factorial(n):
            return 1 if n == 0 else n * factorial(n-1)""",
                    "fibonacci": """def fib(n):
        a, b = 0, 1
        for _ in range(n):
            print(a)
            a, b = b, a + b""",
                "calculator": """def add(a, b):
        return a + b

    def sub(a, b):
        return a - b

    print(add(5, 3))""",
                "sort a list": """def sort_list(lst):
        return sorted(lst)""",
                "reverse a string": """def reverse_string(s):
        return s[::-1]""",
                "check palindrome": """def is_palindrome(s):
        return s == s[::-1]""",
                "find maximum in list": """def find_max(lst):
        return max(lst)""",
                "sum of a list": """def sum_list(lst):
        return sum(lst)""",
                "check if prime": """def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True""",
                "generate random number": """import random
    def random_number(min_val, max_val):
        return random.randint(min_val, max_val)""",
                "find minimum in list": """def find_min(lst):
        return min(lst)""",
                "convert to uppercase": """def to_uppercase(s):
        return s.upper()""",
                "calculate area of circle": """import math
    def circle_area(radius):
        return math.pi * radius ** 2""",  # Fixed missing comma
                "file reader": """def read_file(filename):
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "File not found."
    print(read_file('example.txt'))""",
                "web scraper": """import requests
    from bs4 import BeautifulSoup
    def scrape_title(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.title.string
        except:
            return "Failed to scrape."
    print(scrape_title('https://example.com'))""",
                "simple gui": """import tkinter as tk
    def create_window():
        root = tk.Tk()
        root.title("Aura GUI")
        label = tk.Label(root, text="Hello from Aura!")
        label.pack()
        root.mainloop()
    create_window()""",
                "password generator": """import random
    import string
    def generate_password(length):
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choice(chars) for _ in range(length))
    print(generate_password(12))  # Example: 'K#9mPx!qL2$n'"""
            }
        
            best_match = process.extractOne(prompt.lower(), fallback_code.keys(), scorer=fuzz.partial_ratio)
            if best_match and best_match[1] > 70:
                key = best_match[0]
                code_block = fallback_code[key]
                print(f"\nGenerated code for '{key}':\n```python\n{code_block}\n```")
                with open("generated_code.py", "w") as f:
                    f.write(code_block)
                return f"Code generated for '{key}'. It’s available in the console or 'generated_code.py'. Would you like an explanation?"
            else:
                available_options = ", ".join(f"'{key}'" for key in fallback_code.keys())
                return f"Unable to match your request. Please try one of these: {available_options}."
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            return "An error occurred during code generation. Please try a simpler request."
    
    def execute_code(self, code):
        """Execute the generated code."""
        try:
            exec(code, globals())
            return "Code executed successfully."
        except Exception as e:
            logging.error(f"Code execution error: {e}")
            return "An error occurred during code execution. Please try again."
            
            
            best_match = process.extractOne(prompt.lower(), fallback_code.keys(), scorer=fuzz.partial_ratio)
            if best_match and best_match[1] > 70:
                key = best_match[0]
                code_block = fallback_code[key]
                print(f"\nGenerated code for '{key}':\n```python\n{code_block}\n```")
                with open("generated_code.py", "w") as f:
                    f.write(code_block)
                return f"Code generated for '{key}'. It’s available in the console or 'generated_code.py'. Would you like an explanation?"
            else:
                available_options = ", ".join(f"'{key}'" for key in fallback_code.keys())
                return f"Unable to match your request. Please try one of these: {available_options}."
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            return "An error occurred during code generation. Please try a simpler request."

    def provide_health_tips(self, command):
        """Offer health tips with a professional tone."""
        command = command.lower().replace("health", "").replace("tips", "").strip()
        categories = {
            "stress": ["stress", "anxiety", "worry"],
            "energy": ["energy", "tired", "fatigue"],
            "focus": ["focus", "concentration"],
            "well-being": ["wellness", "wellbeing", "general"],
            "sleep": ["sleep", "insomnia"],
            "anxiety": ["anxiety", "panic"]
        }
        
        best_score = 0
        best_category = "well-being"
        for cat, keywords in categories.items():
            for kw in keywords:
                score = fuzz.ratio(command, kw)
                if score > best_score:
                    best_score = score
                    best_category = cat
        
        tip = random.choice(self.futuristic_tips[best_category])
        return f"Here’s a health tip: {tip}. Does this assist you?"

    def solve_math(self, command):
        """Solve math problems with a professional response."""
        try:
            problem_match = re.search(r"(calculate|math|solve|compute)\s+(.+)", command, re.IGNORECASE)
            if not problem_match:
                return "Please provide a math problem, such as 'calculate 5 + 3' or 'solve x + 2 = 5'."
            problem = problem_match.group(2).strip()
            
            if any(op in problem for op in ['=', 'x', 'y', '^']):
                x = symbols('x')
                problem = problem.replace('^', '**')
                if '=' in problem:
                    lhs, rhs = problem.split('=', 1)
                    expr = sympify(f"Eq({lhs.strip()}, {rhs.strip()})")
                else:
                    expr = sympify(problem)
                solutions = solve(expr, x)
                return f"The solution is: {solutions}. Would you like another calculation?"
            else:
                problem = problem.replace('plus', '+').replace('minus', '-').replace('times', '*').replace('divided by', '/')
                result = sympify(problem)
                return f"The result is: {result.evalf()}. Do you need further assistance?"
        except Exception as e:
            logging.error(f"Math error: {e}")
            return f"An error occurred: {str(e)}. Please try a simpler problem, such as '2 + 3'."

    def get_weather(self, command):
        """Fetch weather with a professional forecast."""
        try:
            city_match = re.search(r"(weather\s+(in\s+)?|what.?s\s+the\s+weather(?:\s+like)?\s+(?:in\s+)?|temperature\s+(?:in\s+)?)(.+)", command, re.IGNORECASE)
            if not city_match:
                return "Please specify a location, such as 'weather in Mumbai'."
            city = city_match.group(2).strip()
            api_key = "d2726712af441a2233527240e370ff57"  # Replace with your valid OpenWeatherMap API key (free tier available)
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return "Weather data is currently unavailable for this location. Please try another city."
            
            data = response.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"].capitalize()
            return f"The weather in {city} is {desc} with a temperature of {temp}°C. Would you like a forecast for tomorrow?"
        except Exception as e:
            logging.error(f"Weather error: {e}")
            return "Weather information is unavailable at this time. Please try again later."

    def set_reminder(self, command):
        """Set a reminder with a professional confirmation."""
        try:
            match = re.search(r"(set|make|add)\s+(?:a\s+)?reminder\s+(?:in\s+)?(\d+)\s+minutes?\s+(?:to\s+)?(.+)", command.lower())
            if not match:
                return "Please specify a reminder, such as 'set a reminder in 5 minutes to call someone'."
            _, minutes, message = match.groups()
            minutes = int(minutes)
            reminder_time = time.time() + (minutes * 60)
            self.reminders.append((reminder_time, message))
            threading.Thread(target=self._check_reminders, daemon=True).start()
            return f"Reminder set for {minutes} minutes from now: {message}. Anything else I can assist with?"
        except Exception as e:
            logging.error(f"Reminder error: {e}")
            return "Unable to set the reminder. Please try again."

    def _check_reminders(self):
        """Check and announce reminders professionally."""
        while self.reminders:
            current_time = time.time()
            for reminder in self.reminders[:]:
                reminder_time, message = reminder
                if current_time >= reminder_time:
                    self.speak(f"Reminder: It’s time to {message}.")
                    self.reminders.remove(reminder)
            time.sleep(1)

    def set_custom_greeting(self, command):
        """Set a custom greeting with a professional response."""
        try:
            match = re.search(r"(set|change)\s+greeting\s+to\s+(.+)", command.lower())
            if match:
                self.custom_greeting = match.group(2)
                self.speak(f"Custom greeting set to: {self.custom_greeting}. It will be used in future interactions.")
            else:
                self.speak("Please specify a greeting, such as 'set greeting to Hello, I’m your assistant'.")
        except Exception as e:
            logging.error(f"Custom greeting error: {e}")
            return "Unable to set the custom greeting. Please try again."

    def listen_to_voice(self, language=None):
        """Listen to voice input with professional handling."""
        if not self.microphone or not self.recognizer:
            return "mic_error"
        with self.microphone as source:
            for attempt in range(2):
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    logging.info("Listening for your command...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    command = self.recognizer.recognize_google(audio, language=self.recognition_lang)
                    logging.info(f"Recognized: {command}")
                    return command.lower()
                except sr.WaitTimeoutError:
                    logging.info("No speech detected.")
                    return "silence"
                except sr.UnknownValueError:
                    if attempt == 0:
                        logging.info("Unintelligible audio, retrying...")
                        continue
                    logging.info("Speech unintelligible after retry.")
                    return "unintelligible"
                except sr.RequestError as e:
                    logging.error(f"Speech recognition service error: {e}")
                    return "speech_service_down"
                except Exception as e:
                    logging.error(f"Audio error: {e}")
                    return "audio_error"
        return "unintelligible"

    def start_interaction(self, wake_word="aura"):
        """Begin interaction with a professional introduction."""
        logging.info(f"Starting interaction with wake word: {wake_word}")
        self.guessing_number = None
        self.speak(f"Greetings. I’m Aura, your virtual assistant. Say '{wake_word}' to activate me, or provide a command directly.")
        idle_count = 0
        
        while True:
            try:
                command = self.listen_to_voice()
                logging.info(f"Raw command: {command}")
                
                if command == "silence":
                    idle_count += 1
                    if idle_count > 2:
                        self.speak(random.choice([
                            "No input detected recently. How may I assist you?",
                            "It’s been quiet. Would you like to proceed with a task?"
                        ]))
                        idle_count = 0
                    continue
                elif command in ["mic_error", "speech_service_down", "audio_error"]:
                    self.speak("An issue occurred with audio input. Please try again.")
                    continue
                elif command == "unintelligible":
                    self.speak("I couldn’t understand your command. Please repeat it clearly.")
                    continue
                
                command = command.strip()
                if self.translator and self.detected_lang == "en" and "set language" not in command.lower():
                    try:
                        detected = self.translator.detect(command).lang
                        if detected in ['hi', 'te', 'ta', 'ml', 'kn', 'es'] and detected != self.speak_lang:
                            self.detected_lang = detected
                            self.speak(f"Detected language: {detected}. Say 'set language to {detected}' to switch, or continue in your current language.")
                            continue
                    except Exception as e:
                        logging.error(f"Language detection failed: {e}")
                        self.detected_lang = "en"
                
                raw_command = command
                if wake_word in command.lower():
                    command = command.lower().replace(wake_word, "").strip()
                    logging.info(f"Wake word detected, processing: {command}")
                else:
                    logging.info(f"No wake word, processing anyway: {command}")

                if "exit" in command or "quit" in command:
                    self.speak(random.choice([
                        "Shutting down now. Goodbye.",
                        "Terminating session. Thank you for using Aura."
                    ]))
                    time.sleep(1)
                    break
                
                response = self.process_command(command, raw_command)
                if response:
                    if "translate" in command.lower() and self.translator:
                        match = re.search(r"translate\s+(.+?)\s+(?:to|into|in)\s+(.+)", command.lower())
                        if match:
                            _, target_lang = match.groups()
                            target_lang = target_lang.strip().lower()
                            lang_codes = {
                                "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta",
                                "malayalam": "ml", "kannada": "kn", "spanish": "es", "french": "fr"
                            }
                            target_code = lang_codes.get(target_lang, target_lang[:2])
                            translated_response = self.translator.translate(response, dest=target_code).text
                            self.speak(translated_response, force_lang=target_code)
                        else:
                            self.speak(response)
                    elif self.speak_lang != "en":
                        try:
                            logging.info(f"Translating response to {self.speak_lang}: {response[:50]}...")
                            translated_response = self.translator.translate(response, dest=self.speak_lang).text
                            self.speak(translated_response, force_lang=self.speak_lang)
                        except Exception as e:
                            logging.error(f"Response translation failed: {e}")
                            self.speak(response)
                    else:
                        self.speak(response)
                idle_count = 0
                
            except Exception as e:
                logging.error(f"Interaction error: {e}")
                self.speak("An error occurred. Please try your command again.")
                time.sleep(1)

    def process_command(self, command, raw_command):
        """Process commands with a professional tone, using raw_command for specific matches."""
        if not isinstance(command, str) or not command:
            return random.choice([
                "No command detected. How may I assist you?",
                "Input was not recognized. Please provide a command."
            ])

        command = self.correct_typos(command)
        self.memory.append(command)
        
        sentiment = {"label": "NEUTRAL", "score": 0.5}
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(command)[0]
                self.user_mood = ("happy" if sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.8 else
                                 "sad" if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.9 else "neutral")
                logging.info(f"Mood detected: {self.user_mood}")
            except Exception:
                logging.warning("Sentiment analysis failed.")

        if self.current_game == "guessing":
            if re.search(r"guess\s+\d+", command):
                return self.guessing_game(command)
            elif "stop" in command or "exit" in command:
                self.current_game = None
                self.guessing_number = None
                return "Guessing game terminated. What would you like to do next?"
            else:
                return "Currently in guessing mode. Please say 'guess [number]' or 'stop game'."
        elif self.current_game == "trivia":
            if re.search(r"answer\s+\d+", command):
                return self.trivia_game(command)
            elif "stop" in command or "exit" in command:
                self.current_game = None
                self.trivia_question = None
                self.trivia_correct_answer = None
                self.trivia_options = []
                return "Trivia game terminated. What would you like to do next?"
            else:
                return "Currently in trivia mode. Please say 'answer [number]' or 'stop game'."

        creator_followup = self.creator_details(command)
        if creator_followup:
            return creator_followup

        responses = {
            r"(who|hu)\s+(are|r)\s+(you|u|is)\s*(aura)?": self.introduce_self,
            r"why\s+(is|was)\s+your\s+name\s*(aura|aura's)?": self.why_name_aura,
            r"translate\s+(.+?)\s+(?:to|into|in)\s+(.+)": self.translate_text,
            r"(set|change)\s+language\s+to\s+(.+)": self.set_language,
            r"(weather\s+(in\s+)?|what.?s\s+the\s+weather(?:\s+like)?\s+(?:in\s+)?|temperature\s+(?:in\s+)?)(.+)": self.get_weather,
            r"(calculate|math|solve|compute)\s+(.+)": self.solve_math,
            r"(generate|write|show\s+me)( a)?\s+code\s*(for|to|on)?\s+(.+)": self.generate_code,
            r"(open|launch|start|run)\s+(.+)": self.open_application,
            r"(detect\s+objects|scan\s+my\s+surroundings|what\s+do\s+you\s+see|analyze\s+my\s+environment)": self.detect_objects_yolo,
            r"(describe\s+my\s+background|what.?s\s+in\s+my\s+background|tell\s+me\s+about\s+my\s+surroundings)": self.describe_background,
            r"(what\s+(?:is\s+the\s+)?time|current\s+time|time\s+now)": self.time_query,
            r"(search|answer|what|why|how|when|where)\s+(.+)": self.answer_question,
            r"(health\s+tips|prescription|how\s+to\s+improve|tips\s+for\s+better)\s+(.+)": self.provide_health_tips,
            r"(news|what.?s\s+happening|latest\s+news)": self.get_news,
            r"(joke|tell\s+me\s+something\s+funny|make\s+me\s+laugh)": self.tell_joke,
            r"(detect\s+emotion|read\s+my\s+mood|how\s+am\s+i\s+feeling)": self.detect_emotion,
            r"(draw|generate|create)\s+(a\s+)?(.+)": self.generate_image,
            r"edit\s+image\s+(resize|rotate|add\s+text)": self._edit_generated_image,  # New pattern for direct editing
            r"(count\s+objects|how\s+many\s+objects|object\s+count)": self.count_objects,
            r"(how\s+are\s+you|how.?s\s+it\s+going|what.?s\s+up)": lambda c: random.choice([
                "I’m fully operational. How can I assist you today?",
                "All systems are functioning. What’s on your agenda?"
            ]),
            r"(tell\s+me\s+more|expand\s+on\s+that|give\s+me\s+details)": self.expand_last_topic,
            r"(predict\s+my\s+future|what.?s\s+in\s+store\s+for\s+me|tell\s+me\s+my\s+fortune)": self.predict_future,
            r"(play\s+guessing\s+game|let.?s\s+play\s+a\s+game|start\s+guessing\s+game)": self.play_guessing_game,
            r"(play\s+trivia|trivia\s+time|start\s+trivia)": self.start_trivia_game,
            r"(set\s+the\s+mood|change\s+the\s+vibe|make\s+it\s+relaxing|make\s+it\s+energetic)": self.set_mood_environment,
            r"guess\s+\d+": self.guessing_game,
            r"(define|meaning\s+of|what\s+is|explain)\s+(.+)": self.define_word,
            r"(test\s+speech|check\s+audio|speech\s+test)": self.test_speech,
            r"(set\s+voice\s+pitch|change\s+voice\s+speed|adjust\s+pitch)\s+(.+)": self.set_voice_pitch,
            r"(set|make|add)\s+(?:a\s+)?reminder\s+(?:in\s+)?(.+)": self.set_reminder,
            r"(set|change)\s+greeting\s+to\s+(.+)": self.set_custom_greeting,
            r"(toggle\s+tts|switch\s+tts\s+engine|change\s+voice\s+engine)": self.toggle_tts_engine,
            r"(who\s+(created|made)\s+(you|aura)|tell\s+me\s+about\s+(mukta|srivatsav|mukka\s+srivatsav|creator))": self.about_creator,
            r"(recognize\s+face|who\s+is\s+this|identify\s+person)": self.recognize_face,
            r"(read\s+text|what\s+is\s+written|read\s+what\s+is\s+on\s+the\s+object)": self.read_text_from_object
        }

        matched_response = None
        for pattern, func in responses.items():
            if re.search(pattern, raw_command, re.IGNORECASE):
                matched_response = func(raw_command)
                break
            elif re.search(pattern, command, re.IGNORECASE):
                matched_response = func(command)
                break

        if matched_response and isinstance(matched_response, str):
            # Adjust response based on detected emotion
            if self.current_emotion in ["happy", "surprise"]:
                matched_response += " I see you’re feeling positive. What’s next?"
            elif self.current_emotion in ["sad", "angry", "fear"]:
                matched_response += " I detect some unease. How can I support you further?"
            elif self.user_mood == "happy":
                matched_response += " I’m glad you’re in good spirits. What’s next?"
            elif self.user_mood == "sad":
                matched_response += " I’m here to assist. How can I support you further?"
            return matched_response

        if self.user_mood == "sad" or self.current_emotion in ["sad", "angry"]:
            return self.handle_negative_sentiment(command)
        return self.fallback_response(command) + " What would you like to do now?"

    def about_creator(self, command):
        """Provide information about the creator professionally."""
        creator_info = (
            "I was created by Mukka Sri Vatsav. He was born on November 30, 2005, and is currently in his second year of BTech, "
            "studying Artificial Intelligence and Machine Learning at Sphoorthy Engineering College. "
            "Would you like additional details about him?"
        )
        return creator_info

    def answer_question(self, command):
        """Answer questions with a professional response."""
        query = re.sub(r"(search\s+for|answer|what|why|how|who|when|where)", "", command, flags=re.IGNORECASE).strip()
        if not query:
            return "Please provide a question or topic for me to address."
        context = " ".join(self.memory[-3:])
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            response = requests.get(url, timeout=5).json()
            if response["query"]["search"]:
                title = response["query"]["search"][0]["title"]
                detail_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={title}&format=json"
                detail_response = requests.get(detail_url, timeout=5).json()
                page = next(iter(detail_response["query"]["pages"].values()))
                if "extract" in page:
                    return f"Here’s some information: {page['extract'][:200]}{'...' if len(page['extract']) > 200 else ''}. Would you like more details?"
        except Exception as e:
            logging.error(f"Wikipedia failed: {e}")
        web_response = self.web_scrape(query)
        if "I couldn’t find" not in web_response:
            return f"Based on web data: {web_response}. Do you have additional questions?"
        if self.chatbot:
            try:
                ai_response = self.chatbot(f"Context: {context}\nQuestion: {query}\nAnswer:", max_length=150, truncation=True)[0]["generated_text"]
                return f"My analysis suggests: {ai_response.split('Answer:')[-1].strip()}. Does this answer your query?"
            except Exception as e:
                logging.error(f"AI response error: {e}")
        return "I couldn’t find an answer to your question. Please try another inquiry."

    def web_scrape(self, query):
        """Scrape the web with a professional approach."""
        try:
            url = f"https://www.google.com/search?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            result = soup.find("div", class_="BNeawe") or soup.find("span", class_="hgKElc")
            return result.text if result else "I couldn’t find relevant information."
        except Exception as e:
            logging.error(f"Web scrape failed: {e}")
            return "I couldn’t find relevant information."

    def translate_text(self, command):
        """Translate text and speak only the translated result."""
        if not self.translator:
            return "Translation services are currently unavailable. Please try again later."
        try:
            match = re.search(r"translate\s+(.+?)\s+(?:to|into|in)\s+(.+)", command, re.IGNORECASE)
            if not match:
                return "Please provide a translation request, such as 'translate hello to Hindi'."
            text, target_lang = match.groups()
            target_lang = target_lang.strip().lower()
            lang_codes = {
                "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta",
                "malayalam": "ml", "kannada": "kn", "spanish": "es", "french": "fr"
            }
            target_code = lang_codes.get(target_lang, target_lang[:2])
            logging.info(f"Translating '{text}' to {target_lang} (code: {target_code})")
            translated = self.translator.translate(text.strip(), dest=target_code).text
            return translated
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return "An error occurred during translation. Please try again."

    def get_news(self, command=None):
        """Fetch news with a professional update."""
        try:
            feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")  # Free BBC News RSS
            if feed.entries:
                return f"Latest news: {feed.entries[0]['title']} - {feed.entries[0]['summary']}. Would you like more updates?"
            return "No news available at this time."
        except Exception as e:
            logging.error(f"News fetch error: {e}")
            return "News retrieval failed. Please try again later."

    def tell_joke(self, command=None):
        """Share a joke with a professional tone."""
        try:
            response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5).json()  # Free joke API
            return f"Here’s a joke: {response['setup']} ... {response['punchline']}. Would you like another?"
        except Exception as e:
            logging.error(f"Joke fetch error: {e}")
            return "Here’s a joke: Why don’t skeletons fight? They lack the guts. Would you like another?"

    def detect_objects_yolo(self, command=None):
        """Detect objects with a professional response."""
        if not self.object_detector:
            return "Object detection is currently unavailable due to a technical issue."
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Unable to access the camera. Please ensure it’s connected."
        logging.info("Detecting objects...")
        
        objects = set()
        start_time = time.time()
        try:
            while cap.isOpened() and time.time() - start_time < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.object_detector(frame, conf=0.3)
                frame_objects = set()
                for result in results:
                    for box in result.boxes:
                        label = self.object_detector.names[int(box.cls[0])]
                        frame_objects.add(label)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                objects.update(frame_objects)
                cv2.imshow("Aura Vision (Press 'q' to quit)", frame)
                if frame_objects and random.random() < 0.3:
                    self.speak(f"Currently detecting: {', '.join(sorted(frame_objects))}.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logging.error(f"Detection error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        final_response = f"Objects detected: {', '.join(sorted(objects))}. Review completed." if objects else "No objects detected in the current view."
        self.speak(final_response)
        return "Object detection initiated. Please view the results on the screen."

    def describe_background(self, command=None):
        """Describe the background with a professional summary."""
        if not self.object_detector:
            return "Background analysis is unavailable due to a technical issue."
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Camera access failed. Please check your device."
        
        objects = set()
        start_time = time.time()
        while time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return "Unable to capture an image. Please try again."
            results = self.object_detector(frame, conf=0.3)
            frame_objects = set(self.object_detector.names[int(box.cls[0])] for result in results for box in result.boxes)
            objects.update(frame_objects)
        
        cap.release()
        response = f"Your background contains: {', '.join(sorted(objects))}. Any specific details you’d like to explore?" if objects else "Your background appears clear of identifiable objects."
        self.speak(response)
        return response

    def recognize_face(self, command=None):
        """Recognize faces in real-time using face_recognition with a celebrity database."""
        if not self.celebrity_db:
            self._load_celebrity_database()  # Ensure database is loaded
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Unable to access the camera. Please ensure it’s connected."
        logging.info("Starting real-time face recognition with celebrity database...")
        
        recognized_names = set()
        start_time = time.time()
        try:
            while cap.isOpened() and time.time() - start_time < 30:  # 30-second scan
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if not face_encodings:
                    cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                        best_match = "Unknown"
                        best_similarity = -1
                        threshold = 0.6  # Confidence threshold
                        for name, celeb_embedding in self.celebrity_db.items():
                            if celeb_embedding is None:
                                continue
                            similarity = self._cosine_similarity(face_encoding, celeb_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = name
                        
                        label = f"{best_match} ({best_similarity:.2f})" if best_similarity >= threshold else "Unknown"
                        if best_similarity >= threshold:
                            recognized_names.add(best_match)
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.imshow("Aura Face Recognition (Press 'q' to quit)", frame)
                if recognized_names and random.random() < 0.3:
                    self.speak(f"Recognized: {', '.join(sorted(recognized_names))}.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        final_response = f"Recognized individuals: {', '.join(sorted(recognized_names))}. Recognition completed." if recognized_names else "No recognizable faces detected."
        self.speak(final_response)
        return "Face recognition initiated. Please view the results on the screen."

    def read_text_from_object(self, command=None):
        """Read text from objects using Tesseract OCR."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Unable to access the camera. Please ensure it’s connected."
        logging.info("Starting text reading from objects...")
        
        detected_text = ""
        try:
            ret, frame = cap.read()
            if not ret:
                return "Unable to capture an image. Please try again."
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            detected_text = pytesseract.image_to_string(gray).strip()
            
            if detected_text:
                cv2.imshow("Text Detection", frame)
                cv2.waitKey(3000)
            else:
                cv2.imshow("Text Detection", frame)
                cv2.waitKey(1000)
        except Exception as e:
            logging.error(f"Text reading error: {e}")
            detected_text = "Unable to read text due to an error."
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        response = f"The text detected is: {detected_text}. Would you like me to read something else?" if detected_text else "No text detected on the object."
        self.speak(response)
        return response

    def time_query(self, command=None):
        """Tell the time with a professional response."""
        try:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            return f"The current time is {now}. How may I assist you further?"
        except Exception:
            return "Unable to retrieve the current time."

    def open_application(self, command):
        """Launch applications with a professional confirmation."""
        app = re.sub(r"(open|launch|start|run)", "", command, flags=re.IGNORECASE).strip()
        apps = {"notepad": "notepad.exe", "chrome": "chrome.exe", "spotify": "spotify.exe"}
        if app in apps:
            try:
                subprocess.Popen([apps[app]])
                return f"Opening {app} now. It should launch momentarily."
            except Exception as e:
                logging.error(f"App launch error: {e}")
                return f"Failed to open {app}. Please ensure it’s installed."
        return f"Application '{app}' is not recognized. Supported options include 'notepad', 'chrome', or 'spotify'. Please try again."

    def handle_negative_sentiment(self, command):
        """Offer support with a professional tone."""
        return random.choice([
            "I detect a lower mood. Would you like a suggestion to improve it, such as a breathing exercise?",
            "It seems you may be feeling down. How can I assist you further?"
        ])

    def expand_last_topic(self, command=None):
        """Expand on the last topic with a professional response."""
        if not self.memory:
            return "No previous topics recorded. What would you like to discuss?"
        last_command = self.memory[-2] if len(self.memory) > 1 else self.memory[-1]
        return self.answer_question(f"tell me more about {last_command}")

    def predict_future(self, command=None):
        """Provide a future prediction with a professional tone."""
        futures = [
            "I predict a productive day ahead. A significant achievement may be forthcoming.",
            "My analysis suggests a positive event soon. Stay prepared for success.",
            "A pleasant development may occur shortly, based on current trends."
        ]
        return random.choice(futures)
    
    def play_guessing_game(self, command=None):
        """Start a guessing game with a structured response."""
        if self.current_game:
            return "A game is already in progress. Say 'stop game' to start a new one."
        self.current_game = "guessing"
        self.guessing_number = random.randint(1, 10)
        return "Guessing game initiated. I’ve selected a number between 1 and 10. Please say 'guess [number]' to begin."

    def guessing_game(self, command):
        """Handle guessing game responses professionally."""
        if self.current_game != "guessing" or self.guessing_number is None:
            return "No guessing game is active. Say 'play guessing game' to start."
        try:
            guess_match = re.search(r"guess\s+(\d+)", command.lower())
            if not guess_match:
                return "Please say 'guess [number]', such as 'guess 5'."
            guess = int(guess_match.group(1))
            if not 1 <= guess <= 10:
                return "Your guess must be between 1 and 10. Please try again."
            if guess == self.guessing_number:
                self.current_game = None
                self.guessing_number = None
                return "Correct. You’ve identified the number. Would you like to play again?"
            elif guess < self.guessing_number:
                return "Your guess is too low. Please try a higher number."
            else:
                return "Your guess is too high. Please try a lower number."
        except ValueError:
            return "Invalid input. Please say 'guess [number]' with a numeric value."
        except Exception as e:
            logging.error(f"Guessing game error: {e}")
            return "An error occurred in the game. Please try again or restart with 'play guessing game'."

    def start_trivia_game(self, command=None):
        """Start a trivia game with a professional introduction."""
        if self.current_game:
            return "A game is currently active. Say 'stop game' to begin a new one."
        question = self.fetch_trivia_question()
        if not question:
            return "Trivia data is unavailable at this time. Please try again later."
        self.current_game = "trivia"
        self.trivia_question = question
        self.trivia_correct_answer = question["correct_answer"]
        options = question["incorrect_answers"] + [self.trivia_correct_answer]
        random.shuffle(options)
        self.trivia_options = options
        question_text = html.unescape(question["question"])
        option_text = ", ".join([f"Option {i+1}: {html.unescape(opt)}" for i, opt in enumerate(options)])
        return f"Trivia game initiated. Question: {question_text}. {option_text}. Please respond with 'answer [number]'."

    def fetch_trivia_question(self):
        """Fetch a trivia question efficiently."""
        try:
            response = requests.get("https://opentdb.com/api.php?amount=1&type=multiple", timeout=5)  # Free Open Trivia DB
            data = response.json()
            if data["response_code"] == 0:
                return data["results"][0]
            else:
                logging.error(f"Trivia API response code: {data['response_code']}")
                return None
        except Exception as e:
            logging.error(f"Trivia fetch error: {e}")
            return None

    def trivia_game(self, command):
        """Process trivia answers with a structured response."""
        if self.current_game != "trivia" or not self.trivia_question:
            return "No trivia game is active. Say 'play trivia' to begin."
        try:
            answer_match = re.search(r"answer\s+(\d+)", command.lower())
            if not answer_match:
                return "Please say 'answer [number]', such as 'answer 1'."
            choice = int(answer_match.group(1)) - 1
            if 0 <= choice < 4:
                selected_option = self.trivia_options[choice]
                if selected_option == self.trivia_correct_answer:
                    response = "Correct. Well done on answering accurately."
                else:
                    response = f"Incorrect. The correct answer is {self.trivia_correct_answer}."
                self.current_game = None
                self.trivia_question = None
                self.trivia_correct_answer = None
                self.trivia_options = []
                return response + " Would you like to play another round? Say 'play trivia'."
            else:
                return "Please select a number between 1 and 4."
        except ValueError:
            return "Invalid input. Please say 'answer [number]' with a numeric value."
        except Exception as e:
            logging.error(f"Trivia game error: {e}")
            return "An error occurred in the trivia game. Please try again with 'play trivia'."

    def set_mood_environment(self, command):
        """Set a mood with a professional suggestion."""
        mood = "relaxing" if "relax" in command.lower() else "energetic" if "energy" in command.lower() else "random"
        if mood == "relaxing":
            return "Imagine a serene holographic sunset over a cyber-ocean. Does this help you relax?"
        elif mood == "energetic":
            return "Picture a vibrant neon cityscape. Ready to feel energized?"
        else:
            return "Consider a zero-gravity environment for a unique experience. How does that sound?"

    def define_word(self, command):
        """Define words with a professional explanation."""
        try:
            word_match = re.search(r"(define|meaning\s+of|what\s+is)\s+(.+)", command, re.IGNORECASE)
            if not word_match:
                return "Please specify a word to define, such as 'define hope'."
            word = word_match.group(2).strip()
            
            for attempt in range(3):
                try:
                    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"  # Free Dictionary API
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    if isinstance(data, list) and data and "meanings" in data[0] and data[0]["meanings"]:
                        meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
                        return f"The definition of '{word}' is: {meaning}. Would you like another definition?"
                    else:
                        logging.error(f"No definition found for '{word}': {data}")
                        return f"No definition found for '{word}'. Please try another word."
                except requests.RequestException as e:
                    logging.error(f"Definition fetch attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return f"Unable to retrieve the definition for '{word}' due to: {str(e)}. Please try again later."
        except Exception as e:
            logging.error(f"Unexpected definition error: {e}")
            return f"An error occurred while defining '{word}': {str(e)}. Please try a different word."

    def correct_typos(self, command):
        """Correct typos with a subtle adjustment, improved for speech-to-text errors."""
        keywords = [
            "who", "are", "you", "why", "is", "your", "name", "aura", "open", "detect", "describe", "time", "weather", 
            "search", "generate", "health", "translate", "news", "calculate", "joke", "exit", "how", "tell", "predict", 
            "play", "set", "guess", "define", "test", "toggle", "created", "about", "recognize", "face", "read", "text"
        ]
        command_words = command.split()
        corrected_words = []
        
        for word in command_words:
            best_match = process.extractOne(word, keywords, scorer=fuzz.WRatio)
            if best_match and best_match[1] > 85:
                corrected_words.append(best_match[0])
            else:
                corrected_words.append(word)
        
        corrected_command = " ".join(corrected_words)
        if corrected_command != command:
            logging.info(f"Corrected '{command}' to '{corrected_command}'")
        
        if "hu" in corrected_command.lower() and "r" in corrected_command.lower() and "u" in corrected_command.lower():
            corrected_command = "who are you"
            logging.info(f"Corrected '{command}' to 'who are you' (special case)")
        
        return corrected_command

    def fallback_response(self, command):
        """Handle unknown commands with a professional fallback."""
        ai_response = self.answer_question(command)
        if "I couldn’t find" not in ai_response:
            return ai_response
        return random.choice([
            f"Unable to process '{command}'. Would you like to try a different command, such as weather or news?",
            f"The command '{command}' is not recognized. Please try an alternative request.",
            f"I cannot interpret '{command}'. May I assist you with something else?"
        ])
    
    def detect_emotion(self, command=None):
        """Detect emotion in real-time using DeepFace and adapt responses."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Unable to access the camera. Please ensure it’s connected."
        logging.info("Starting real-time emotion detection...")
        
        emotions_detected = []
        start_time = time.time()
        try:
            while cap.isOpened() and time.time() - start_time < 10:  # 10-second scan
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = analysis[0]['dominant_emotion'] if analysis else "neutral"
                    emotions_detected.append(dominant_emotion)
                    self.current_emotion = dominant_emotion
                    
                    # Display emotion on frame
                    face_locations = DeepFace.extract_faces(frame, enforce_detection=False)
                    for face in face_locations:
                        facial_area = face["facial_area"]
                        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    logging.warning(f"Emotion detection error: {e}")
                    cv2.putText(frame, "Emotion: Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv2.imshow("Aura Emotion Detection (Press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logging.error(f"Emotion detection error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if emotions_detected:
            most_common_emotion = max(set(emotions_detected), key=emotions_detected.count)
            self.current_emotion = most_common_emotion
            response = f"Detected emotion: {most_common_emotion}. How can I assist you based on this?"
            self.speak(response)
            return response
        else:
            response = "No emotions detected during the scan."
            self.speak(response)
            return response

    def generate_image(self, command):
        """Generate or edit an image based on user command with multiple styles."""
        try:
            match = re.search(r"(draw|generate|create)\s+(a\s+)?(.+)", command.lower())
            if not match:
                return "Please specify what to generate, such as 'generate a futuristic city'."
            description = match.group(3).strip()
        
            self.speak(f"Do you want me to generate an image of '{description}'? Please say 'yes' or 'no'.")
            confirmation = self.listen_to_voice()
            if confirmation and "yes" in confirmation.lower():
                # Determine style based on description
                style = "text"  # Default
                if "futuristic" in description.lower():
                    style = "futuristic"
                elif "abstract" in description.lower():
                    style = "abstract"
            
                # Generate image based on style
                if style == "futuristic":
                    img = self._generate_futuristic_image(description)
                elif style == "abstract":
                    img = self._generate_abstract_image(description)
                else:
                    img = self._generate_text_image(description)
            
                img.save("generated_image.png")
                self.speak("Image generated and saved as 'generated_image.png'. Would you like to edit it? Say 'yes' or 'no'.")
                edit_response = self.listen_to_voice()
                if edit_response and "yes" in edit_response.lower():
                    return self._edit_generated_image("generated_image.png")
                return f"Image of '{description}' generated and saved as 'generated_image.png'."
            else:
                return "Image generation canceled. How else can I assist you?"
        except Exception as e:
            logging.error(f"Image generation error: {e}")
            return "Unable to generate the image. Please try again."

    def _generate_futuristic_image(self, description):
        """Generate a futuristic-style image."""
        img = Image.new('RGB', (400, 400), color='black')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Draw futuristic elements
        draw.rectangle([(50, 50), (350, 350)], fill='darkblue', outline='cyan')
        draw.line([(50, 50), (350, 350)], fill='cyan', width=2)
        draw.line([(50, 350), (350, 50)], fill='cyan', width=2)
        draw.text((100, 180), description, fill='white', font=font)
        return img

    def _generate_abstract_image(self, description):
        """Generate an abstract-style image."""
        img = Image.new('RGB', (400, 400), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Random abstract lines
        for _ in range(10):
            x1, y1 = random.randint(0, 400), random.randint(0, 400)
            x2, y2 = random.randint(0, 400), random.randint(0, 400)
            draw.line([(x1, y1), (x2, y2)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
        draw.text((150, 180), description, fill='black', font=font)
        return img

    def _generate_text_image(self, description):
        """Generate a simple text-based image."""
        img = Image.new('RGB', (400, 400), color='black')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((50, 180), description, fill='white', font=font)
        return img

    def _edit_generated_image(self, filename):
        """Edit a previously generated image."""
        try:
            img = Image.open(filename)
            self.speak("Editing options: 'resize', 'rotate', or 'add text'. Please specify one.")
            edit_command = self.listen_to_voice()
            if not edit_command:
                return "No editing command received."
        
            if "resize" in edit_command.lower():
                img = img.resize((200, 200))
                img.save("edited_image.png")
                return "Image resized to 200x200 and saved as 'edited_image.png'."
            elif "rotate" in edit_command.lower():
                img = img.rotate(90)
                img.save("edited_image.png")
                return "Image rotated 90 degrees and saved as 'edited_image.png'."
            elif "add text" in edit_command.lower():
                self.speak("What text would you like to add?")
                text = self.listen_to_voice()
                if text and text != "unintelligible":
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.load_default()
                    draw.text((50, 50), text, fill='red', font=font)
                    img.save("edited_image.png")
                    return f"Text '{text}' added to image and saved as 'edited_image.png'."
                return "No valid text provided for editing."
            else:
                return "Editing option not recognized. Options are 'resize', 'rotate', or 'add text'."
        except Exception as e:
            logging.error(f"Image editing error: {e}")
            return "Failed to edit the image. Please try again."

    def count_objects(self, command=None):
        """Count objects in the environment using YOLO."""
        if not self.object_detector:
            return "Object detection is unavailable due to a technical issue."
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Unable to access the camera. Please ensure it’s connected."
        logging.info("Counting objects...")
        
        object_counts = {}
        start_time = time.time()
        try:
            while cap.isOpened() and time.time() - start_time < 10:  # 10-second scan
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.object_detector(frame, conf=0.3)
                frame_counts = {}
                for result in results:
                    for box in result.boxes:
                        label = self.object_detector.names[int(box.cls[0])]
                        frame_counts[label] = frame_counts.get(label, 0) + 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {frame_counts[label]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for label, count in frame_counts.items():
                    object_counts[label] = max(object_counts.get(label, 0), count)
                cv2.imshow("Aura Object Count (Press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logging.error(f"Object counting error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if object_counts:
            count_str = ", ".join(f"{label}: {count}" for label, count in object_counts.items())
            response = f"Objects counted: {count_str}. Would you like a detailed breakdown?"
            self.speak(response)
            return response
        else:
            response = "No objects detected to count."
            self.speak(response)
            return response

if __name__ == "__main__":
    aura = AuraAI()
    aura.start_interaction()