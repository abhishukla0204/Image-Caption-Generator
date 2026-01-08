# Image Caption Generator Using Deep Learning

An end-to-end image captioning system using CNN (for image feature extraction) and LSTM (for caption generation). Upload any image and the model generates a natural language description.


## Overview

This project combines Computer Vision and Natural Language Processing to automatically generate captions for images. The architecture uses:

- **CNN Encoder**: Extracts visual features from images using a pretrained network (VGG16/ResNet50/DenseNet201)
- **LSTM Decoder**: Generates captions word-by-word using the image features and previously predicted words

## Dataset

The model is trained on the **Flickr8k** dataset from Kaggle:  
🔗 [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- **8,000 images** with **5 captions each** (40,000 image-caption pairs)
- Diverse scenes including people, animals, activities, and objects

## Project Structure

```
├── flickr8k-image-captioning-using-cnns-lstms.ipynb   # Training notebook
├── main.py                      # Streamlit inference app
├── models/
│   ├── model.keras              # Trained caption model
│   ├── feature_extractor.keras  # CNN feature extractor
│   └── tokenizer.pkl            # Fitted tokenizer
├── img.png                      # Demo images
├── img_1.png
├── img_2.png
├── img_3.png
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Image-Caption-Generator-Using-Deep-Learning.git
   cd Image-Caption-Generator-Using-Deep-Learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (for training)
   - Download from [Kaggle Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Extract to your preferred location
   - Update paths in the notebook accordingly

## Usage

### Run the Streamlit App

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501`, upload an image, and get a caption!

### Train Your Own Model

1. Open `flickr8k-image-captioning-using-cnns-lstms.ipynb` in Jupyter/VS Code
2. Update the dataset paths to your local Flickr8k location
3. Run all cells to train the model
4. Save the trained model, feature extractor, and tokenizer to the `models/` folder

## Model Architecture

```
Image → CNN (Feature Extractor) → Image Embedding (1920-dim)
                                          ↓
Caption → Tokenizer → Embedding → LSTM → Dense → Next Word
                                   ↑
                          [Image Embedding]
```

1. **Image Features**: Pretrained CNN extracts a feature vector from the input image
2. **Text Preprocessing**: Captions are lowercased, cleaned, and wrapped with `startseq`/`endseq` tokens
3. **Training**: The model learns to predict the next word given the image features and previous words
4. **Inference**: Starting with `startseq`, the model generates words until `endseq` or max length

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas
- Matplotlib, Seaborn


## Acknowledgments

- Dataset: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) by Aditya Jain
- Inspired by the "Show and Tell" paper by Vinyals et al.

## Author

**Abhinav Shukla**

---

⭐ If you found this project helpful, please give it a star!