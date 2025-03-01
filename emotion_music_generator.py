import cv2
import numpy as np
import pygame
import pygame.midi
import time
import threading
from pygame.locals import *
import random
import sys


class EmotionMusicGenerator:
    def __init__(self):
        # UI elements - define these first so they're available during initialization
        self.screen_width = 1024
        self.screen_height = 768
        self.webcam_display_size = (480, 360)  # Larger webcam display
        self.particles = []

        # Camera orientation settings
        self.rotation_modes = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        self.current_rotation = 0  # Start with no rotation

        # Application state
        self.running = True
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.emotion_history = []
        self.emotion_colors = {
            "happy": (255, 255, 0),  # Yellow
            "sad": (0, 0, 255),  # Blue
            "angry": (255, 0, 0),  # Red
            "surprise": (255, 165, 0),  # Orange
            "neutral": (200, 200, 200)  # Gray
        }

        # Emotion values
        self.emotion_values = {
            "happy": 0.1,
            "sad": 0.1,
            "angry": 0.1,
            "surprise": 0.1,
            "neutral": 0.6
        }

        # Emotion smoothing for stable detection
        self.emotion_smoothing = {
            "happy": 0.1,
            "sad": 0.1,
            "angry": 0.1,
            "surprise": 0.1,
            "neutral": 0.6
        }
        self.smoothing_factor = 0.3  # Lower = more smoothing

        # Initialize components
        self.initialize_pygame()
        self.initialize_webcam()
        self.initialize_music_generator()

        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.webcam_processing_loop)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()

        # Start music thread
        self.music_thread = threading.Thread(target=self.music_generation_loop)
        self.music_thread.daemon = True
        self.music_thread.start()

    def initialize_pygame(self):
        """Initialize Pygame and its modules"""
        pygame.init()
        pygame.midi.init()
        pygame.display.set_caption("Emotion-Responsive Music Generator")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

    def initialize_webcam(self):
        """Set up webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()

        # Set resolution to 720p for better face detection
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Set additional camera properties for better quality
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Slightly increase brightness

        self.current_frame = None

    def initialize_music_generator(self):
        """Set up the music generator with MIDI output"""
        try:
            # List available MIDI devices for debugging
            print("Available MIDI devices:")
            device_count = pygame.midi.get_count()
            if device_count == 0:
                print("No MIDI devices found!")
            else:
                for i in range(device_count):
                    print(f"Device {i}: {pygame.midi.get_device_info(i)}")

            # Try to get default MIDI device id
            try:
                default_id = pygame.midi.get_default_output_id()
                if default_id != -1:
                    self.midi_out = pygame.midi.Output(default_id)
                    print(f"MIDI output initialized successfully using device {default_id}")
                else:
                    print("No default MIDI output device found")
                    self.midi_out = None
            except Exception as e:
                print(f"MIDI default device error: {e}")
                self.midi_out = None

            self.current_notes = []

            # Define musical parameters for each emotion
            self.emotion_music_params = {
                "happy": {
                    "scale": [0, 2, 4, 5, 7, 9, 11],  # Major scale
                    "base_note": 60,  # C4
                    "tempo": 120,
                    "velocity": 100,
                    "note_length": 0.2
                },
                "sad": {
                    "scale": [0, 2, 3, 5, 7, 8, 10],  # Minor scale
                    "base_note": 60,
                    "tempo": 80,
                    "velocity": 70,
                    "note_length": 0.4
                },
                "angry": {
                    "scale": [0, 1, 4, 5, 7, 8, 11],  # Phrygian dominant
                    "base_note": 48,
                    "tempo": 140,
                    "velocity": 120,
                    "note_length": 0.1
                },
                "surprise": {
                    "scale": [0, 2, 4, 6, 8, 10],  # Whole tone
                    "base_note": 72,
                    "tempo": 100,
                    "velocity": 90,
                    "note_length": 0.15
                },
                "neutral": {
                    "scale": [0, 2, 4, 5, 7, 9, 11],  # Major scale
                    "base_note": 60,
                    "tempo": 100,
                    "velocity": 80,
                    "note_length": 0.3
                }
            }
        except Exception as e:
            print(f"MIDI initialization error: {e}")
            print("Music generation will be simulated only.")
            self.midi_out = None

    def webcam_processing_loop(self):
        """Process webcam frames and detect emotions using DeepFace"""
        # Try to import DeepFace
        try:
            from deepface import DeepFace
            use_deepface = True
            print("DeepFace imported successfully, using real emotion detection.")
        except ImportError:
            use_deepface = False
            print("DeepFace not found. Using simulated emotions only.")
            print("Install DeepFace with: pip install deepface")

        # Initialize variables
        last_analysis_time = 0
        analysis_interval = 1.0  # Analyze once per second to reduce CPU usage
        last_emotion_update = time.time()

        # Create a face cascade for faster detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Debug camera settings
        print(
            f"Camera resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # Main webcam loop
        while self.running:
            try:
                # Get frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame from camera")
                    time.sleep(0.5)
                    continue

                # Process the frame
                frame = cv2.flip(frame, 1)  # Flip horizontally for selfie view

                # Apply current rotation setting
                if self.rotation_modes[self.current_rotation] is not None:
                    frame = cv2.rotate(frame, self.rotation_modes[self.current_rotation])

                current_time = time.time()

                # Emotion detection with DeepFace if available
                if use_deepface:
                    # Try multiple detection methods if needed
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Method 1: Standard Haar cascade
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,  # Even more sensitive (was 1.1)
                        minNeighbors=1,  # Minimum threshold
                        minSize=(30, 30),  # Minimum face size
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # If no faces found, try with equalized histogram for better contrast
                    if len(faces) == 0:
                        equalized = cv2.equalizeHist(gray)
                        faces = face_cascade.detectMultiScale(
                            equalized,
                            scaleFactor=1.05,
                            minNeighbors=1,
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )

                    # If still no faces found with Haar cascade, try a different cascade
                    if len(faces) == 0:
                        # Try profile face detector
                        profile_face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_profileface.xml')
                        if not profile_face_cascade.empty():
                            faces = profile_face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.05,
                                minNeighbors=1
                            )

                    # Log if still no faces detected after all attempts
                    if len(faces) == 0 and current_time - last_analysis_time > 2.0:
                        print("Trying alternate camera orientation...")
                        # Flip the frame vertically to try a different orientation
                        frame = cv2.flip(frame, 0)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=1,
                            minSize=(20, 20),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        # If still no faces, print debug info
                        if len(faces) == 0:
                            print("No face detected. Check lighting and camera position.")
                            # Save debug frame to help diagnose
                            cv2.imwrite("debug_frame.jpg", frame)
                            print("Saved debug frame to 'debug_frame.jpg'")
                        last_analysis_time = current_time

                    # If faces found and it's time to analyze
                    if len(faces) > 0 and current_time - last_analysis_time > analysis_interval:
                        try:
                            # For DeepFace, try direct detection without relying on Haar cascade
                            if len(faces) == 0:
                                print("Trying direct DeepFace detection...")
                                # Try direct DeepFace detection without pre-filtering
                                try:
                                    direct_analysis = DeepFace.analyze(
                                        frame,
                                        actions=['emotion'],
                                        enforce_detection=True,  # Force detection
                                        detector_backend="opencv",  # Try different backends
                                        silent=True
                                    )

                                    if direct_analysis and len(direct_analysis) > 0:
                                        # Extract face region from DeepFace detection
                                        region = direct_analysis[0]['region']
                                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                                        faces = np.array([[x, y, w, h]])
                                        print("Face detected directly by DeepFace!")
                                except Exception as e:
                                    print(f"Direct DeepFace detection failed: {e}")

                            # Get the largest face
                            if len(faces) > 0:


                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Extract face region for analysis
                            face_region = frame[y:y + h, x:x + w]

                            # Only analyze if face region is valid
                            if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                                # Analyze emotions
                                analysis = DeepFace.analyze(
                                    face_region,
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    silent=True
                                )

                                # Process analysis results
                                if analysis and len(analysis) > 0:
                                    emotions = analysis[0]['emotion']

                                    # Map emotions to our format
                                    new_emotion_values = {
                                        "happy": emotions['happy'] / 100.0,
                                        "sad": emotions['sad'] / 100.0,
                                        "angry": emotions['angry'] / 100.0,
                                        "surprise": emotions['surprise'] / 100.0,
                                        "neutral": emotions['neutral'] / 100.0
                                    }

                                    # Apply smoothing
                                    for emotion in new_emotion_values:
                                        if emotion in self.emotion_smoothing:
                                            new_emotion_values[emotion] = (
                                                    self.smoothing_factor * new_emotion_values[emotion] +
                                                    (1 - self.smoothing_factor) * self.emotion_smoothing[emotion]
                                            )
                                            self.emotion_smoothing[emotion] = new_emotion_values[emotion]

                                    # Update current emotion
                                    self.emotion_values = new_emotion_values
                                    dominant_emotion = max(self.emotion_values.items(), key=lambda x: x[1])
                                    self.current_emotion = dominant_emotion[0]
                                    self.emotion_intensity = dominant_emotion[1]

                                    # Record in history
                                    if len(self.emotion_history) > 100:
                                        self.emotion_history.pop(0)
                                    self.emotion_history.append((self.current_emotion, self.emotion_intensity))

                                    # Show emotion on frame
                                    emotion_text = f"{self.current_emotion.capitalize()}: {self.emotion_intensity:.2f}"
                                    cv2.putText(
                                        frame,
                                        emotion_text,
                                        (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        self.emotion_colors.get(self.current_emotion, (200, 200, 200)),
                                        2
                                    )

                            last_analysis_time = current_time

                        except Exception as e:
                            print(f"Emotion analysis error: {e}")

                    # Draw all detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            self.emotion_colors.get(self.current_emotion, (200, 200, 200)),
                            2
                        )

                # Simulated emotion detection when DeepFace isn't available
                else:
                    if current_time - last_emotion_update > 3.0:
                        # Simulate changing emotions
                        emotions = self.emotion_values.copy()
                        for emotion in emotions:
                            change = random.uniform(-0.1, 0.1)
                            emotions[emotion] = max(0.1, min(0.9, emotions[emotion] + change))

                        # Normalize
                        total = sum(emotions.values())
                        for emotion in emotions:
                            emotions[emotion] /= total

                        # Update current emotion
                        self.emotion_values = emotions
                        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                        self.current_emotion = dominant_emotion[0]
                        self.emotion_intensity = dominant_emotion[1]

                        # Record in history
                        if len(self.emotion_history) > 100:
                            self.emotion_history.pop(0)
                        self.emotion_history.append((self.current_emotion, self.emotion_intensity))

                        last_emotion_update = current_time

                    # Add visual indicator for current emotion
                    color = self.emotion_colors.get(self.current_emotion, (200, 200, 200))
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
                    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

                    # Draw face placeholder
                    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                    cv2.rectangle(
                        frame,
                        (center_x - 100, center_y - 100),
                        (center_x + 100, center_y + 100),
                        color,
                        2
                    )

                    # Show current emotion
                    cv2.putText(
                        frame,
                        f"{self.current_emotion.capitalize()}",
                        (center_x - 90, center_y - 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                # Convert frame to RGB for Pygame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame

                # Sleep to reduce CPU load
                time.sleep(0.03)

            except Exception as e:
                print(f"Error in webcam processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

    def music_generation_loop(self):
        """Generate music based on detected emotions"""
        last_note_time = 0

        while self.running:
            # Check if emotion_music_params is defined
            if not hasattr(self, 'emotion_music_params'):
                time.sleep(0.1)
                continue

            # Get the current emotion parameters
            current_params = self.emotion_music_params.get(self.current_emotion,
                                                           self.emotion_music_params["neutral"])

            current_time = time.time()
            note_interval = 60 / current_params["tempo"]

            # Check if it's time to play a new note
            if current_time - last_note_time > current_params["note_length"]:
                # Stop all currently playing notes
                if self.midi_out:
                    for note in self.current_notes:
                        self.midi_out.note_off(note, 0, 0)
                self.current_notes = []

                # Play a new note or chord based on the emotion
                if self.midi_out:
                    # Choose a note from the scale
                    scale = current_params["scale"]
                    base_note = current_params["base_note"]

                    # Simple algorithmic composition
                    if self.current_emotion == "happy":
                        # For happy, play major triads
                        root = base_note + random.choice(scale)
                        third = root + 4  # Major third
                        fifth = root + 7  # Perfect fifth
                        self.midi_out.note_on(root, current_params["velocity"], 0)
                        self.midi_out.note_on(third, current_params["velocity"], 0)
                        self.midi_out.note_on(fifth, current_params["velocity"], 0)
                        self.current_notes = [root, third, fifth]
                    elif self.current_emotion == "sad":
                        # For sad, play minor triads
                        root = base_note + random.choice(scale)
                        third = root + 3  # Minor third
                        fifth = root + 7  # Perfect fifth
                        self.midi_out.note_on(root, current_params["velocity"], 0)
                        self.midi_out.note_on(third, current_params["velocity"], 0)
                        self.midi_out.note_on(fifth, current_params["velocity"], 0)
                        self.current_notes = [root, third, fifth]
                    elif self.current_emotion == "angry":
                        # For angry, play diminished chords with staccato
                        root = base_note + random.choice(scale)
                        third = root + 3  # Minor third
                        fifth = root + 6  # Diminished fifth
                        self.midi_out.note_on(root, current_params["velocity"], 0)
                        self.midi_out.note_on(third, current_params["velocity"], 0)
                        self.midi_out.note_on(fifth, current_params["velocity"], 0)
                        self.current_notes = [root, third, fifth]
                    else:
                        # For other emotions, play simple melody notes
                        note = base_note + random.choice(scale)
                        self.midi_out.note_on(note, current_params["velocity"], 0)
                        self.current_notes = [note]

                # Add visual particles for the notes
                for _ in range(3):
                    self.add_particle()

                last_note_time = current_time

            # Sleep to reduce CPU load
            time.sleep(0.01)

    def add_particle(self):
        """Add a music visualization particle"""
        color = self.emotion_colors.get(self.current_emotion, (200, 200, 200))
        # Randomize starting position on the right side of the screen
        x = self.screen_width - 50
        y = random.randint(self.screen_height // 2, self.screen_height - 100)
        # Randomize speed and size based on emotion intensity
        speed = -2 - (self.emotion_intensity * 5)
        size = 5 + int(self.emotion_intensity * 15)
        lifespan = 100 + int(self.emotion_intensity * 100)

        self.particles.append({
            "pos": [x, y],
            "velocity": [speed, random.uniform(-1, 1)],
            "color": color,
            "size": size,
            "life": lifespan
        })

    def update_particles(self):
        """Update and remove particles"""
        for particle in self.particles[:]:
            # Update position
            particle["pos"][0] += particle["velocity"][0]
            particle["pos"][1] += particle["velocity"][1]

            # Update life
            particle["life"] -= 1

            # Remove if out of bounds or dead
            if (particle["pos"][0] < 0 or
                    particle["pos"][0] > self.screen_width or
                    particle["pos"][1] < 0 or
                    particle["pos"][1] > self.screen_height or
                    particle["life"] <= 0):
                self.particles.remove(particle)

    def render_ui(self):
        """Render the user interface"""
        # Fill background
        self.screen.fill((30, 30, 30))

        # Draw title
        title = self.font.render("Emotion-Responsive Music Generator", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))

        # Draw webcam feed if available
        if self.current_frame is not None:
            frame_surface = pygame.surfarray.make_surface(self.current_frame)
            frame_surface = pygame.transform.scale(frame_surface, self.webcam_display_size)
            self.screen.blit(frame_surface, (20, 60))

        # Draw emotion info
        y_pos = 60
        # Draw current emotion
        emotion_text = self.font.render(
            f"Current Emotion: {self.current_emotion.capitalize()} ({self.emotion_intensity:.2f})",
            True, self.emotion_colors.get(self.current_emotion, (200, 200, 200)))
        self.screen.blit(emotion_text, (360, y_pos))
        y_pos += 40

        # Draw emotion bars
        for emotion, value in self.emotion_values.items():
            bar_width = int(value * 200)
            emotion_name = emotion.capitalize()
            text = self.font.render(f"{emotion_name}:", True, (255, 255, 255))
            self.screen.blit(text, (360, y_pos))

            # Draw emotion bar
            pygame.draw.rect(self.screen, self.emotion_colors.get(emotion, (200, 200, 200)),
                             (470, y_pos + 5, bar_width, 20))
            pygame.draw.rect(self.screen, (100, 100, 100),
                             (470, y_pos + 5, 200, 20), 1)

            y_pos += 30

        # Draw music parameters
        y_pos += 20
        params_title = self.font.render("Music Parameters:", True, (255, 255, 255))
        self.screen.blit(params_title, (360, y_pos))
        y_pos += 30

        current_params = self.emotion_music_params.get(self.current_emotion, self.emotion_music_params["neutral"])

        tempo_text = self.font.render(f"Tempo: {current_params['tempo']} BPM", True, (255, 255, 255))
        self.screen.blit(tempo_text, (360, y_pos))
        y_pos += 30

        scale_name = "Major" if self.current_emotion in ["happy",
                                                         "neutral"] else "Minor" if self.current_emotion == "sad" else "Custom"
        scale_text = self.font.render(f"Scale: {scale_name}", True, (255, 255, 255))
        self.screen.blit(scale_text, (360, y_pos))
        y_pos += 30

        intensity_text = self.font.render(f"Intensity: {current_params['velocity']}", True, (255, 255, 255))
        self.screen.blit(intensity_text, (360, y_pos))

        # Draw emotion history graph
        graph_x = 20
        graph_y = 320
        graph_width = self.screen_width - 40
        graph_height = 150

        pygame.draw.rect(self.screen, (50, 50, 50), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, (100, 100, 100), (graph_x, graph_y, graph_width, graph_height), 1)

        # Draw graph title
        graph_title = self.font.render("Emotion History", True, (255, 255, 255))
        self.screen.blit(graph_title, (graph_x, graph_y - 30))

        # Draw emotion history
        if self.emotion_history:
            # Calculate points for each line
            points_per_emotion = {emotion: [] for emotion in self.emotion_colors.keys()}

            for i, (emotion, intensity) in enumerate(self.emotion_history):
                x = graph_x + (i / len(self.emotion_history)) * graph_width
                y = graph_y + graph_height - (intensity * graph_height)
                points_per_emotion[emotion].append((x, y))

            # Draw lines for each emotion
            for emotion, points in points_per_emotion.items():
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.emotion_colors.get(emotion), False, points, 2)

        # Draw particles
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle["color"],
                               (int(particle["pos"][0]), int(particle["pos"][1])),
                               particle["size"])

        # Draw keyboard controls
        y_pos = self.screen_height - 180
        controls_title = self.font.render("Manual Emotion Controls:", True, (255, 255, 255))
        self.screen.blit(controls_title, (20, y_pos))
        y_pos += 30

        controls = [
            "H - Set emotion to Happy",
            "S - Set emotion to Sad",
            "A - Set emotion to Angry",
            "P - Set emotion to Surprise",
            "N - Set emotion to Neutral",
            "R - Rotate camera view"
        ]

        for control in controls:
            text = pygame.font.SysFont("Arial", 16).render(control, True, (180, 180, 180))
            self.screen.blit(text, (20, y_pos))
            y_pos += 20

        # Draw usage instructions
        instructions = [
            "Press ESC to exit"
        ]

        y_pos = self.screen_height - 40
        for instruction in instructions:
            text = pygame.font.SysFont("Arial", 16).render(instruction, True, (180, 180, 180))
            self.screen.blit(text, (20, y_pos))
            y_pos += 20

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False

            # Manual emotion controls
            elif event.type == KEYDOWN:
                if event.key == K_h:  # Happy
                    self.set_emotion("happy", 0.8)
                elif event.key == K_s:  # Sad
                    self.set_emotion("sad", 0.8)
                elif event.key == K_a:  # Angry
                    self.set_emotion("angry", 0.8)
                elif event.key == K_p:  # Surprise
                    self.set_emotion("surprise", 0.8)
                elif event.key == K_n:  # Neutral
                    self.set_emotion("neutral", 0.8)
                elif event.key == K_r:  # Rotate camera
                    self.current_rotation = (self.current_rotation + 1) % len(self.rotation_modes)
                    rotation_name = "None" if self.rotation_modes[self.current_rotation] is None else \
                        "90° CW" if self.rotation_modes[self.current_rotation] == cv2.ROTATE_90_CLOCKWISE else \
                            "180°" if self.rotation_modes[self.current_rotation] == cv2.ROTATE_180 else \
                                "90° CCW"
                    print(f"Camera rotation changed to: {rotation_name}")

    def set_emotion(self, emotion, intensity):
        """Manually set the current emotion"""
        self.current_emotion = emotion
        self.emotion_intensity = intensity

        # Update emotion values
        for e in self.emotion_values:
            self.emotion_values[e] = 0.1
        self.emotion_values[emotion] = intensity

        # Normalize
        total = sum(self.emotion_values.values())
        for e in self.emotion_values:
            self.emotion_values[e] /= total

        # Reset smoothing
        for e in self.emotion_smoothing:
            self.emotion_smoothing[e] = self.emotion_values.get(e, 0.1)

        # Add to history
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)
        self.emotion_history.append((emotion, intensity))

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'midi_out') and self.midi_out:
            self.midi_out.close()
        pygame.midi.quit()
        pygame.quit()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Main application loop"""
        try:
            while self.running:
                self.handle_events()
                self.update_particles()
                self.render_ui()

                pygame.display.flip()
                self.clock.tick(30)

        finally:
            self.cleanup()


def main():
    print("Starting Emotion-Responsive Music Generator...")
    print("This application will attempt to use DeepFace for real emotion detection if available.")
    print("If DeepFace is not installed, it will fall back to simulated emotions.")
    print("\nManual emotion controls are always available:")
    print("  H - Happy")
    print("  S - Sad")
    print("  A - Angry")
    print("  P - Surprise")
    print("  N - Neutral")
    print("\nCamera controls:")
    print("  R - Rotate camera view (cycle through orientations)")
    print("\nConnect a MIDI output device for music generation.")
    print("Press ESC to exit the application.\n")

    try:
        app = EmotionMusicGenerator()
        app.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # Keep the console open


if __name__ == "__main__":
    main()