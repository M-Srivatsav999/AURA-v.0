# AURA-v.0
AuraAI: An advanced, open-source virtual assistant with voice, vision, and creative capabilities. Built with free tools, featuring real-time object detection, face recognition, image generation and many..
# AuraAI - Augmented Utility Reality Assistant

**AuraAI** is a sophisticated voice-driven virtual assistant created by Mukka Srivatsav, a second-year BTech student in Artificial Intelligence and Machine Learning at Sphoorthy Engineering College. Built entirely with free tools, it showcases advanced AI capabilities including computer vision, natural language processing, and creative utilities. This project is shared for viewing purposes only—copying, modification, or distribution is strictly prohibited.

Inspired by fictional assistants like JARVIS, AuraAI (Augmented Utility Reality Assistant) offers a glimpse into futuristic AI, designed as a personal project to explore cutting-edge technology.

---

## Features

### Voice Interaction
- **Speech Recognition**: Captures commands via `speech_recognition` (Google Speech API).
- **Text-to-Speech**: Responds with `pyttsx3` (Microsoft Zira) or `gTTS` for multilingual support.
- **Multilingual**: Supports English, Hindi, Telugu, Tamil, Malayalam, Kannada, and Spanish (`googletrans`).
- **Customizable**: Adjustable voice pitch and TTS engine switching.

### Computer Vision
- **Object Detection**: Real-time object identification with YOLOv8 (`ultralytics`) and OpenCV.
- **Face Recognition**: Matches faces against a database of 68 celebrities using `face_recognition`.
- **Emotion Detection**: Analyzes mood with `deepface`.
- **Text Reading**: Extracts text from objects via `pytesseract`.

### Creative Tools
- **Image Generation**: Creates images (`PIL`) with editing options (resize, rotate, add text).
- **Code Generation**: Produces Python scripts (e.g., factorial, sorting) saved to `generated_code.py`.

### Utility Functions
- **Knowledge**: Answers questions (Wikipedia, web scraping), defines words (`DictionaryAPI`), fetches news (BBC RSS).
- **Math Solver**: Solves equations with `sympy`.
- **Weather**: Real-time updates via OpenWeatherMap (API key required).
- **Reminders**: Voice-based timed alerts.
- **Games**: Guessing and trivia (`OpenTriviaDB`).
- **Apps**: Opens tools like Notepad or Chrome.

### Personality
- Professional tone: "Greetings. I’m Aura, your virtual assistant."
- Mood-adaptive responses and futuristic health tips.

---

## Tech Stack
- **Python**: 3.8+.
- **Vision**: `cv2`, `ultralytics`, `face_recognition`, `deepface`, `pytesseract`.
- **Voice**: `speech_recognition`, `pyttsx3`, `gtts`, `pygame`.
- **NLP**: `transformers` (GPT-Neo), `googletrans`, `fuzzywuzzy`.
- **Utilities**: `requests`, `bs4`, `feedparser`, `sympy`, `numpy`, `PIL`.
- **Other**: `threading`, `queue`, `pickle`, `logging`.

---

## Installation (For Reference Only)
*Note*: This is for informational purposes—running or modifying the code is not permitted.

### Prerequisites
- Python 3.8+.
- Webcam and microphone.
- Tesseract OCR (path: `pytesseract.pytesseract.tesseract_cmd`).
- Internet for APIs.

### Steps
1. Clone: `git clone https://github.com/[your-username]/AuraAI.git`
2. Install: `pip install -r requirements.txt`
3. Tesseract: Install and update path.
4. Run: `python aura_ai.py` (viewing only, no usage rights).

---

## Usage Examples
- "Aura, who are you?" - Introduction.
- "Recognize face" - Celebrity matching.
- "Detect objects" - Scans surroundings.
- "Generate code to sort a list" - Script creation.
- "Solve 2x = 8" - Math solution.

---

## Requirements.txt
opencv-python
ultralytics
transformers
pyttsx3
speechrecognition
fuzzywuzzy
feedparser
requests
beautifulsoup4
gtts
pygame
sympy
googletrans==3.1.0a0
pytesseract
numpy
deepface
pillow
face_recognition



---

## Copyright Notice
© 2025 Mukka Srivatsav. All rights reserved. This code is provided for viewing purposes only. Copying, modifying, distributing, or using this software in any form without explicit written permission from the author is strictly prohibited.

---

## About the Creator
Mukka Srivatsav, born November 30, 2005, is a second-year BTech student in AI and ML at Sphoorthy Engineering College.
