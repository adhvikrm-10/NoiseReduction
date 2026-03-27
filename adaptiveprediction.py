import os
import numpy as np
from scipy.io import wavfile

muf = 0.0001
num_tapsf = 128
percent = 0.001

def adaptive_predictor_lms(signal, mu, num_taps, delay=10):
    n = len(signal)
    weights = np.zeros(num_taps)
    # The reference is the signal itself, shifted by 'delay'
    ref = np.zeros(n)
    ref[delay:] = signal[:-delay]
    output_speech = np.zeros(n)
    
    for i in range(num_taps, n):
        # Input vector from the delayed reference
        x = ref[i:i-num_taps:-1]
        # FIR Filter prediction
        prediction = np.dot(weights, x)
        # Error calculation
        # In ALE, the 'prediction' is the correlated speech
        # The 'error' is the uncorrelated noise
        e = signal[i] - prediction
        # Weight update
        weights = weights + 2 * mu * e * x
        output_speech[i] = prediction # We keep the 'predicted' part
        
    return output_speech

def spectral_subtraction(signal, samplerate):
    # Perform Fast Fourier Transform
    spec = np.fft.rfft(signal)
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    # Assume the first 5% of the audio is just noise (silence)
    noise_estimate = np.mean(magnitude[:int(len(magnitude)*percent)])
    
    # Subtract noise magnitude from the signal magnitude
    # We use np.maximum to ensure we don't get negative values
    cleaned_magnitude = np.maximum(magnitude - (noise_estimate * 2.0), 0)
    
    # Reconstruct the signal using original phase and cleaned magnitude
    cleaned_spec = cleaned_magnitude * np.exp(1j * phase)
    return np.fft.irfft(cleaned_spec)



def select_wav_file(folder_path):
    # 1. Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        return None

    # 2. Get all .wav files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not files:
        print("No .wav files found in this folder.")
        return None

    # 3. Display the files to the user
    print(f"\n--- Available Files in '{folder_path}' ---")
    for idx, filename in enumerate(files):
        print(f"[{idx}] {filename}")

    # 4. Let the user choose
    try:
        choice = int(input("\nEnter the number of the file you want to process: "))
        if 0 <= choice < len(files):
            selected_file = os.path.join(folder_path, files[choice])
            print(f"Selected: {files[choice]}")
            return selected_file
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Please enter a valid number.")
        return None


def normalize_audio(data):
    # 1. Determine the maximum absolute value (the peak)
    peak = np.max(np.abs(data))
    # 2. Avoid division by zero if the file is silent
    if peak == 0:
        return data
    # 3. Scale the data to the range [-1.0, 1.0]
    normalized_data = data / peak
    
    return normalized_data

# --- Configuration ---
TARGET_FOLDER = "D:\\signal project\\train_10dB\\10dB\\"  # Change this to your folder path

# --- Execution ---

while (True):
    file_path = select_wav_file(TARGET_FOLDER)
    if file_path:
        samplerate, data = wavfile.read(file_path)
        # Ensure it's mono (some LMS filters struggle with 2D stereo arrays)
        if len(data.shape) > 1:
            data = data[:, 0]

        print(f"Successfully loaded {file_path} at {samplerate}Hz")
        signalf = normalize_audio(data)
        signalf =  adaptive_predictor_lms(signalf, muf, num_tapsf)
        output = spectral_subtraction(signalf, samplerate)
        target_directory = os.path.dirname(file_path)
        save_path = os.path.join(target_directory, "output.wav")

        # 2. Convert float back to 16-bit integer (PCM)
        # We use 32767 to avoid clipping at the absolute limit
        output_int16 = (output * 32767).astype(np.int16)
        # 3. Save the file
        wavfile.write(save_path, samplerate, output_int16)
        print(f"File successfully saved to: {save_path}")
        # Now you can pass 'data' into your adaptive_predictor_lms function!
