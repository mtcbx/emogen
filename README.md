# Emogen

## A simple Flask web app that generates Apple-like emoji images.

This project loads a **fine-tuned diffusion model**, trained on a dataset of **Apple emojis** that was augmented and preprocessed.

---

## ✨ Features

- **Web UI** to enter a prompt.
- Generates **Apple-like emoji** images.
- Uses **AI** (diffusion model).
- Automatically selects the best available compute device:
  - **CUDA** (NVIDIA GPU)
  - **MPS** (Apple Silicon GPU/Metal)
  - CPU fallback

---

## 🧠 Fine-Tuned Model

The model used in this project is located in: `fine_tuned_model`.

It is a **fine-tuned version of `table-diffusion-v1-4`**, trained on a emoji dataset available on Kaggle [here](https://www.kaggle.com/datasets/subinium/emojiimage-dataset). During fine-tuning, only the U-Net weights (`pipeline.unet`) were updated.

### Dataset

The training dataset originally contained emoji images from multiple companies (Apple, Google, Samsung etc.), and was:

- **filtered** to keep only **Apple emojis**
- **cleaned**:
  - removed unwanted tokens from textual descriptions
  - normalized text formatting (punctuation, capitalization)
- **decoded**:
  - images were originally stored in Base64 and were decoded before training
- **augmented using a language model**
  - same emoji images duplicated with multiple different textual descriptions to improve robustness and prompt understanding
- **preprocessed** before training:
  - image resizing
  - generation of `(image, text)` pairs
  - conversion to tensors
  - normalization

---

## 📦 Installation

### 1) Clone the repo

```bash
git clone https://github.com/mtcbx/emogen.git
cd emogen
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 Usage

### 1) Start the Flask app
Run the application with:
```bash
python app.py
```
You should see something like: `Running on http://127.0.0.1:5000`.

### 2) Open a browser
Go to: `http://127.0.0.1:5000`.

### 3) Generate an emoji image
1. Enter a text prompt (e.g. smiling face).
2. Click the Generate button.
3. Wait while the image is being generated (this can take some time).
4. The generated image will be displayed on the page and saved to
`static/generated/generated_image.png`.

Note that each new generation overwrites the previous 'generated_image.png'.
