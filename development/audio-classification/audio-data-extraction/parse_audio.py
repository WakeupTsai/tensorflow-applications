import glob
import librosa
import numpy as np
import pandas as pd


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


def parse_audio_files(filenames):
    features, labels = np.zeros((1, 193)), np.zeros((1, 10))
    i = 0
    try:
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(filenames)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except:
        print(filenames)
    else:
        # One Hot
        features[i] = ext_features
        i += 1
    return features, labels

if __name__ == '__main__':
    X, y = parse_audio_files('audio-min-samples/siren.wav')
    np.savez('audio-min-samples/siren', X=X, y=y)
