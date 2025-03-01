# Emotion-Responsive Music Generator

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen)

A real-time system that analyzes your facial expressions, determines your emotional state, and generates custom music that matches or complements your mood. The interface visually represents your emotions and demonstrates how they directly influence musical elements.

## Features

- **Real-time Emotion Detection**: Uses your webcam and facial analysis to detect emotions (happy, sad, angry, surprise, neutral)
- **Emotion-Driven Music Generation**: Creates algorithmic musical compositions that adapt to your emotional state
- **Dynamic Visualization**: Shows the relationship between your emotions and music parameters
- **Interactive Interface**: Includes emotion history graph, parameter controls, and visual particle effects
- **MIDI Output**: Generates MIDI messages for high-quality sound output with any MIDI device or software synth

## Requirements

- Python 3.7+
- Webcam
- MIDI output device or software synthesizer (optional)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/emotion-music-generator.git
cd emotion-music-generator
```

2. **Set up a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install opencv-python pygame numpy deepface
```

Note: DeepFace will automatically install additional dependencies like TensorFlow.

4. **Set up MIDI output (optional)**

- **Windows**: Install [VirtualMIDISynth](https://coolsoft.altervista.org/en/virtualmidisynth) or use any MIDI-compatible software
- **macOS**: Use GarageBand, Logic, or the built-in IAC Driver
- **Linux**: Install FluidSynth: `sudo apt-get install fluidsynth`

## Usage

Run the application:

```bash
python emotion_music_generator.py
```

The application will:
1. Access your webcam
2. Start analyzing your facial expressions
3. Generate music based on detected emotions
4. Visualize the emotion-to-music relationship

## How It Works

### Emotion Detection
The system uses DeepFace (a deep learning facial analysis framework) to detect emotions from webcam input. It analyzes your facial expressions and classifies them into five emotional categories:
- Happy
- Sad
- Angry
- Surprise
- Neutral

### Music Generation
Each emotion is mapped to specific musical parameters:
- **Happy**: Major scales with bright chords and upbeat tempo
- **Sad**: Minor scales with soft, slower progressions
- **Angry**: Dissonant patterns with intense, rapid notes
- **Surprise**: Unexpected chord progressions with dynamic variations
- **Neutral**: Balanced, moderate musical patterns

### Visualization
The interface includes:
- Live webcam feed with emotional analysis overlay
- Emotion intensity bars
- Musical parameter visualization
- Flowing particle effects that respond to music
- Emotion history graph

## Controls

### Keyboard Controls
- **H**: Set emotion to Happy
- **S**: Set emotion to Sad
- **A**: Set emotion to Angry
- **P**: Set emotion to Surprise
- **N**: Set emotion to Neutral
- **R**: Rotate camera view (cycle through orientations)
- **ESC**: Exit application

## Advanced Configuration

You can modify these parameters in the code:
- Emotion-to-music mappings
- Visualization settings
- Camera properties
- MIDI settings

## Troubleshooting

### Face Detection Issues
- Ensure good lighting on your face
- Position yourself directly in front of the camera
- Use the R key to rotate the camera view to the correct orientation
- Check your webcam is working properly

### MIDI Sound Issues
- Verify a MIDI output device is selected in your system
- Try a different MIDI output device
- Check volume levels in your system

### Performance Problems
- Close other applications using the webcam
- Reduce the camera resolution in the code if needed
- Update your graphics drivers

## Future Development

Planned features:
- User profiles to save preferences
- More sophisticated music generation algorithms
- Recording and exporting capabilities
- Multi-person emotion detection
- Customizable sound palettes for each emotion

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- OpenCV for computer vision capabilities
- DeepFace for facial emotion recognition
- Pygame for the interface and MIDI functionality

---
