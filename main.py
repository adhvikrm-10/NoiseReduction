import os
import re
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows, lfilter

LMS_MU = 0.005
LMS_TAPS = 32
LMS_DELAY = 10

RLS_DELTA = 100.0
RLS_LAMBDA = 0.998
RLS_TAPS = 32

FRAME_SIZE = 512
HOP_SIZE = 256
SS_ALPHA = 1.5
SS_FLOOR = 0.02

WIENER_ALPHA = 0.98
MMSE_ALPHA = 0.98

CLEAN_FOLDER = "D:\\signal project\\clean\\"
TARGET_FOLDER = "D:\\signal project\\train_10dB\\10dB\\"

def method_lms(signal, mu=LMS_MU, num_taps=LMS_TAPS, delay=LMS_DELAY):
    n = len(signal)
    w = np.zeros(num_taps)
    ref = np.zeros(n)
    ref[delay:] = signal[:-delay]
    out = np.zeros(n)

    for i in range(num_taps, n):
        x = ref[i:i - num_taps:-1]
        pred = np.dot(w, x)
        e = signal[i] - pred
        w += 2 * mu * e * x
        out[i] = e

    return out

def method_rls(signal, delta=RLS_DELTA, lam=RLS_LAMBDA,
               num_taps=RLS_TAPS, delay=LMS_DELAY):
    n = len(signal)
    w = np.zeros(num_taps)
    P = np.eye(num_taps) * delta
    ref = np.zeros(n)
    ref[delay:] = signal[:-delay]
    out = np.zeros(n)

    for i in range(num_taps, n):
        x = ref[i:i - num_taps:-1]
        pred = float(np.dot(w, x))
        e = signal[i] - pred

        Px = P @ x
        k = Px / (lam + float(np.dot(x, Px)))
        w = w + k * e
        P = (P - np.outer(k, x) @ P) / lam

        out[i] = e

    return out

def method_spectral_subtraction(signal, samplerate,
                                frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                                alpha=SS_ALPHA, floor=SS_FLOOR):
    win = windows.hann(frame_size)
    n = len(signal)
    output = np.zeros(n + frame_size)

    mags = [np.abs(np.fft.rfft(signal[s:s + frame_size] * win))
            for s in range(0, n - frame_size + 1, hop_size)]
    noise_mag = np.min(np.array(mags), axis=0)

    for start in range(0, n - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size] * win
        spec = np.fft.rfft(frame)
        mag, phase = np.abs(spec), np.angle(spec)
        clean_mag = np.maximum(mag - alpha * noise_mag, floor * mag)
        output[start:start + frame_size] += np.fft.irfft(clean_mag * np.exp(1j * phase)) * win

    return output[:n]

def method_wiener(signal, samplerate,
                  frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                  alpha=WIENER_ALPHA):
    win = windows.hann(frame_size)
    n = len(signal)
    output = np.zeros(n + frame_size)
    n_bins = frame_size // 2 + 1

    noise_frames = min(10, (n - frame_size) // hop_size)
    noise_psd = np.zeros(n_bins)
    for k in range(noise_frames):
        s = k * hop_size
        noise_psd += np.abs(np.fft.rfft(signal[s:s + frame_size] * win)) ** 2
    noise_psd /= max(noise_frames, 1)

    smooth_psd = noise_psd.copy()

    for start in range(0, n - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size] * win
        spec = np.fft.rfft(frame)
        power = np.abs(spec) ** 2
        phase = np.angle(spec)

        smooth_psd = alpha * smooth_psd + (1 - alpha) * power
        snr_post = np.maximum(smooth_psd / (noise_psd + 1e-12) - 1.0, 0.0)

        H = snr_post / (snr_post + 1.0)
        clean_mag = H * np.abs(spec)

        output[start:start + frame_size] += np.fft.irfft(clean_mag * np.exp(1j * phase)) * win

    return output[:n]

def method_mmse(signal, samplerate,
                frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                alpha=MMSE_ALPHA):
    from scipy.special import expn

    win = windows.hann(frame_size)
    n = len(signal)
    output = np.zeros(n + frame_size)
    n_bins = frame_size // 2 + 1

    noise_frames = min(10, (n - frame_size) // hop_size)
    noise_psd = np.zeros(n_bins)
    for k in range(noise_frames):
        s = k * hop_size
        noise_psd += np.abs(np.fft.rfft(signal[s:s + frame_size] * win)) ** 2
    noise_psd /= max(noise_frames, 1)
    noise_psd = np.maximum(noise_psd, 1e-12)

    prev_gain = np.ones(n_bins)
    prev_power = noise_psd.copy()

    for start in range(0, n - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size] * win
        spec = np.fft.rfft(frame)
        power = np.abs(spec) ** 2
        phase = np.angle(spec)

        gamma = np.maximum(power / noise_psd, 1e-12)

        xi = alpha * (prev_gain ** 2) * prev_power / noise_psd \
             + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, 1e-12)

        v = xi * gamma / (xi + 1.0)
        v = np.maximum(v, 1e-12)
        gain = (xi / (xi + 1.0)) * np.exp(0.5 * (-expn(1, v)))
        gain = np.minimum(gain, 1.0)

        noise_psd = alpha * noise_psd + (1 - alpha) * np.minimum(power, noise_psd * 2)

        prev_gain = gain
        prev_power = power

        clean_mag = gain * np.abs(spec)
        output[start:start + frame_size] += np.fft.irfft(clean_mag * np.exp(1j * phase)) * win

    return output[:n]

def compute_snr(clean, degraded):
    min_len = min(len(clean), len(degraded))
    clean = clean[:min_len].astype(np.float64)
    degraded = degraded[:min_len].astype(np.float64)
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean((degraded - clean) ** 2)
    if p_noise == 0:
        return float('inf')
    return 10.0 * np.log10(p_signal / p_noise)

def normalize_audio(data):
    peak = np.max(np.abs(data))
    return data if peak == 0 else data / peak

def extract_clean_name(noisy_filepath):
    base = os.path.splitext(os.path.basename(noisy_filepath))[0]
    m = re.search(r'_clnsp(\d+)$', base)
    if m:
        return f"clnsp{m.group(1)}.wav"
    clean_id = re.split(r'_train|_sn|_babble|_café|_SNR', base, flags=re.IGNORECASE)[0]
    return clean_id + ".wav"

def load_clean_signal(noisy_filepath):
    clean_name = extract_clean_name(noisy_filepath)
    clean_path = os.path.join(CLEAN_FOLDER, clean_name)
    if not os.path.exists(clean_path):
        print(f"  [SNR] Clean file not found: {clean_path}")
        return None, None
    sr, raw = wavfile.read(clean_path)
    if len(raw.shape) > 1:
        raw = raw[:, 0]
    print(f"  [SNR] Matched clean: {clean_name}")
    return sr, normalize_audio(raw.astype(np.float64))

def select_wav_file(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: folder '{folder_path}' not found.")
        return None
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not files:
        print("No .wav files found.")
        return None
    print(f"\n--- Files in '{folder_path}' ---")
    for idx, f in enumerate(files):
        print(f"[{idx}] {f}")
    try:
        choice = int(input("\nEnter file number: "))
        if 0 <= choice < len(files):
            path = os.path.join(folder_path, files[choice])
            print(f"Selected: {files[choice]}")
            return path
    except Exception:
        pass
    print("Invalid selection.")
    return None
