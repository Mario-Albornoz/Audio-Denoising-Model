# Guitar Audio Denoising using Demucs

## Overview

This project implements an AI-based guitar audio denoising system using a **Demucs-based neural network architecture**. The model is designed to remove environmental noise from guitar recordings and reconstruct clean guitar signals suitable for music production, analysis, and signal processing applications.

The system is trained on synthetically generated noisy guitar recordings created by mixing clean guitar signals with environmental noise. The goal is to enable robust denoising across a wide variety of real-world recording conditions.

---

## Model Architecture

The model is based on the **Demucs (Deep Extractor for Music Sources)** architecture, a convolutional encoder–decoder network with skip connections designed for high-quality audio source separation and denoising.

Key characteristics:

- Time-domain audio processing
- Encoder–decoder structure
- Skip connections for detail preservation
- Convolutional layers for feature extraction
- High-quality waveform reconstruction

The architecture is well-suited for guitar denoising because it preserves transient details and harmonic structure while suppressing background noise.

---

## Dataset

### Guitar Recordings

Clean guitar recordings were obtained from the following datasets:

**GuitarSet (Guitar Techniques Dataset):**
https://guitar-techs.github.io/

This dataset provides high-quality recordings of guitar performances across various techniques and playing styles.

---

### Environmental Noise Recordings

Environmental noise samples were obtained from the DEMAND dataset:

https://zenodo.org/records/1227121

The DEMAND dataset provides diverse real-world environmental recordings, including:

- Indoor noise
- Outdoor noise
- Public spaces
- Transportation environments
- Natural environments

---

## Dataset Generation

The training dataset was generated synthetically by mixing clean guitar recordings with environmental noise.

For each training example:

1. A clean guitar recording is selected.
2. An environmental noise sample is selected.
3. The noise signal is scaled to a random Signal-to-Noise Ratio (SNR).
4. The noise is mixed with the guitar recording.
5. The clean guitar signal is stored as the training target.

This produces paired samples:

- **Input:** Noisy guitar recording
- **Target:** Clean guitar recording

This supervised setup allows the model to learn effective noise removal.

---

## Directory Structure

Example project structure:
