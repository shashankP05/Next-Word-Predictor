
#  WikiText-103 Next-Word Prediction Model

This project implements a **Next Word Prediction** model using LSTM trained on the **WikiText-103** dataset. It includes a **Streamlit GUI** for interactive predictions based on user input sentences.

---

##  Features

- Cleaned and tokenized WikiText dataset
- LSTM-based language model built with TensorFlow/Keras
- Top-K word prediction with probabilities
- Interactive Streamlit Web App
- Sample predictions included
- Saves and loads trained model and tokenizer
- Graphs for training/validation loss and accuracy

---

## 🧠 Model Summary

- **Architecture**: Embedding → 2×LSTM → Dense
- **Sequence Length**: 5 previous words
- **Vocabulary Size**: ~10,000 words
- **Dataset**: WikiText-103 (first 50,000 sentences for fast training)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

---

## 🛠️ Installation

### For Training (Google Colab or Local)
```bash
pip install tensorflow transformers datasets matplotlib
```

### For GUI
```bash
pip install streamlit tensorflow numpy
```

---

##  Usage

### 1. 🏋️ Train the Model

Run the training script (`model_training.ipynb` or Python file) to train the model and save:

- `next_word_model.h5`
- `tokenizer.pickle`

### 2. 🌐 Launch the Streamlit GUI

Make sure the model and tokenizer are in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

### 3.  Sample Predictions in Code

```python
from predict import predict_next_word

text = "artificial intelligence is"
print(predict_next_word(text, top_k=3))
# Output: [('useful', 0.45), ('dangerous', 0.30), ('transforming', 0.15)]
```

---

## 🔍 Available Functions

- `predict_next_word(text, top_k=3)` – Predict top-k words
- `predict_next_word_simple(text)` – Get most likely next word
- `interactive_prediction()` – CLI prediction mode
- `load_saved_model()` – Load model and tokenizer

---

##  Training Results

Training and validation loss/accuracy are plotted to evaluate model performance.

---

##  Sample Sentences

Tested examples:

- "once upon a time in"
- "the quick brown fox"
- "artificial intelligence is"
- "deep learning models can"

Top-3 next word predictions shown for each.

---

## 📚 Dataset

Dataset: [WikiText-103](https://huggingface.co/datasets/wikitext)







