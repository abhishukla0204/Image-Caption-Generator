<div align="center">

#  Image Caption Generator Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An end-to-end image captioning system using CNN + LSTM architecture**

*Upload any image → Get a natural language description automatically!*

• [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Dataset](#-dataset)

</div>

---

##  Overview

This project combines **Computer Vision** and **Natural Language Processing** to automatically generate captions for images. The architecture uses:

| Component | Description |
|-----------|-------------|
|  **CNN Encoder** | Extracts visual features using pretrained **DenseNet201** |
|  **LSTM Decoder** | Generates captions word-by-word from image features |
|  **Streamlit App** | Interactive web interface for real-time inference |



---

## 📊 Dataset

The model is trained on the **Flickr8k** dataset from Kaggle:

<div align="center">

[![Kaggle](https://img.shields.io/badge/Dataset-Flickr8k-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/adityajn105/flickr8k)

</div>

| Metric | Value |
|--------|-------|
|  Total Images | 8,000 |
|  Captions per Image | 5 |
|  Total Samples | 40,000 |
|  Content | People, animals, activities, objects |

---

## Project Structure

```
 Image-Caption-Generator
├──  image-captioning-cnns-lstms.ipynb  # Training notebook
├──  main.py                            # Streamlit inference app
├──  models/
│   ├──  model.keras                    # Trained caption model
│   ├──  feature_extractor.keras        # CNN feature extractor
│   └──  tokenizer.pkl                  # Fitted tokenizer
├──  img.png, img_1.png, ...            # Demo images
├──  requirements.txt                   # Dependencies
└──  README.md
```

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/abhinavshukla0022/Image-Caption-Generator-Using-Deep-Learning.git
cd Image-Caption-Generator-Using-Deep-Learning
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download dataset (for training only)

- Download from [Kaggle Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Extract to `kaggle/input/flickr8k/` folder
- Update paths in the notebook if needed

---

## 💻 Usage

### 🎮 Run the Streamlit App

```bash
streamlit run main.py
```

Then open your browser to **http://localhost:8501**, upload an image, and get a caption!

###  Train Your Own Model

1. Open `image-captioning-cnns-lstms.ipynb` in Jupyter/VS Code
2. Ensure dataset paths point to your local Flickr8k location
3. Run all cells to train the model (~1-2 hours on GPU)
4. Models are automatically saved to the `models/` folder

---

##  Model Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Image     │ ──► │  DenseNet201     │ ──► │ Image Embedding │
│  (224×224)  │     │  (CNN Encoder)   │     │   (1920-dim)    │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Caption   │ ──► │   Tokenizer +    │ ──► │  LSTM Decoder   │ ──► Next Word
│  (partial)  │     │   Embedding      │     │  + Dense Layer  │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

### How it works:

1. **Feature Extraction**: DenseNet201 extracts a 1920-dimensional feature vector from the input image
2. **Text Preprocessing**: Captions are lowercased, cleaned, and wrapped with `startseq`/`endseq` tokens
3. **Training**: Model learns to predict the next word given image features + previous words
4. **Inference**: Starting with `startseq`, generates words until `endseq` or max length

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥2.10.0 | Deep learning framework |
| Streamlit | ≥1.20.0 | Web app interface |
| NumPy | ≥1.21.0 | Numerical computing |
| Pandas | ≥1.3.0 | Data manipulation |
| Matplotlib | ≥3.5.0 | Visualization |
| Pillow | ≥9.0.0 | Image processing |

---

## Acknowledgments

- **Dataset**: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) by Aditya Jain
- **Inspiration**: ["Show and Tell: A Neural Image Caption Generator"](https://arxiv.org/abs/1411.4555) by Vinyals et al.
- **Architecture**: CNN-LSTM encoder-decoder framework

---

## 👨‍💻 Author

<div align="center">

**Abhinav Shukla**

Email: [shukla.abhinav0204@gmail.com](mailto:shukla.abhinav0204@gmail.com)

</div>

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

</div>