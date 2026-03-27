# NoiseReduction

Noise Reduction in Speech Signals Using Digital Filters (Adaptive Prediction)

## Overview
This project performs noise reduction on speech signals using a hybrid approach combining:
- Adaptive Line Enhancer (LMS-based prediction)
- Spectral Subtraction (FFT-based)
The system processes `.wav` audio files and outputs a cleaned version of the signal.

## Dataset
NOIZEUS dataset (real-world noisy speech samples)

## Key Features
- LMS Adaptive Predictor (Adaptive Line Enhancer)
- FFT-based Spectral Subtraction
- Automatic normalization of input signal
- Interactive file selection from dataset
- Output saved as cleaned `.wav` file

## Techniques Used

### Adaptive Filtering (LMS Predictor)
- Uses delayed version of the signal as reference
- Predicts correlated speech components
- Removes uncorrelated noise

### Spectral Subtraction
- Applies FFT to convert signal to frequency domain
- Estimates noise from initial segment
- Subtracts noise magnitude and reconstructs signal

## Workflow
Input WAV → Normalize → LMS Adaptive Prediction → Spectral Subtraction → Output WAV

## How It Works
1. User selects a `.wav` file from a folder
2. Audio is converted to mono (if needed)
3. Signal is normalized
4. LMS adaptive filter extracts speech
5. Spectral subtraction removes residual noise
6. Output is saved as `output.wav`





