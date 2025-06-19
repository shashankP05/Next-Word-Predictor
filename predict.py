# # Simple Next-Word Prediction Script
# # Use this script to make predictions with your saved model

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle
# import re

# # Load the saved model and tokenizer
# print("🔄 Loading model and tokenizer...")

# try:
#     # Load BOTH files
#     model = load_model('next_word_model.h5')      # Neural network
#     with open('tokenizer.pkl', 'rb') as f:
#         tokenizer = pickle.load(f)                # Text processor
    
#     print("✅ Model and tokenizer loaded successfully!")
    
# except Exception as e:
#     print(f"❌ Error loading files: {e}")
#     print("Make sure both 'next_word_model.h5' and 'tokenizer.pickle' exist!")
#     exit()

# # Text cleaning function (same as training)
# def clean_text(text):
#     """Clean text the same way as during training"""
#     if not isinstance(text, str) or text.strip() == '':
#         return ""
    
#     text = text.lower()
#     text = re.sub(r'@-@', '-', text)
#     text = re.sub(r'= = .+ = =', '', text)
#     text = re.sub(r'= .+ =', '', text)
#     text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
    
#     return text.strip()

# def predict_next_word(input_text, top_k=3):
#     """
#     Predict next word(s) for given input text
    
#     Args:
#         input_text: String to predict next word for
#         top_k: Number of top predictions to return
    
#     Returns:
#         List of (word, probability) tuples
#     """
#     # Step 1: Clean the input text
#     cleaned_text = clean_text(input_text)
    
#     # Step 2: Convert text to numbers using tokenizer
#     tokens = tokenizer.texts_to_sequences([cleaned_text])[0]
    
#     # Step 3: Take last 5 words (sequence length used in training)
#     seq_length = 5
#     if len(tokens) >= seq_length:
#         input_sequence = tokens[-seq_length:]
#     else:
#         # Pad with zeros if input is shorter
#         input_sequence = [0] * (seq_length - len(tokens)) + tokens
    
#     # Step 4: Reshape for model input
#     input_sequence = np.array([input_sequence])
    
#     # Step 5: Get predictions from model (returns probabilities)
#     predictions = model.predict(input_sequence, verbose=0)[0]
    
#     # Step 6: Get top k predictions
#     top_indices = np.argsort(predictions)[-top_k:][::-1]
    
#     # Step 7: Convert numbers back to words using tokenizer
#     results = []
#     for idx in top_indices:
#         if idx in tokenizer.index_word:
#             word = tokenizer.index_word[idx]
#             probability = predictions[idx]
#             results.append((word, probability))
    
#     return results

# def predict_single_word(input_text):
#     """Get just the most likely next word"""
#     predictions = predict_next_word(input_text, top_k=1)
#     if predictions:
#         return predictions[0][0]
#     return "<unknown>"

# # Test the functions
# print("\n🎯 Testing predictions...")

# test_cases = [
#     "once upon a time in",
#     "the quick brown fox",
#     "artificial intelligence is",
#     "machine learning can",
#     "deep learning models",
#     "natural language processing",
#     "i am going to"
# ]

# for text in test_cases:
#     predictions = predict_next_word(text, top_k=3)
#     print(f"\nInput: '{text}'")
#     print("Predictions:")
#     for i, (word, prob) in enumerate(predictions, 1):
#         print(f"  {i}. '{word}' (confidence: {prob:.1%})")

# # Interactive prediction
# def interactive_mode():
#     """Interactive prediction mode"""
#     print("\n🎮 Interactive Mode - Enter text to get next word predictions!")
#     print("Type 'quit' to exit\n")
    
#     while True:
#         user_input = input("Enter text: ").strip()
        
#         if user_input.lower() == 'quit':
#             break
        
#         if not user_input:
#             print("Please enter some text!")
#             continue
        
#         try:
#             # Get predictions
#             predictions = predict_next_word(user_input, top_k=5)
            
#             if predictions:
#                 print(f"\nNext word predictions for: '{user_input}'")
#                 for i, (word, prob) in enumerate(predictions, 1):
#                     print(f"  {i}. '{word}' (confidence: {prob:.1%})")
#             else:
#                 print("No predictions available.")
                
#         except Exception as e:
#             print(f"Error: {e}")
        
#         print()

# # Display available functions
# print("\n" + "="*50)
# print("🚀 READY TO MAKE PREDICTIONS!")
# print("="*50)
# print("\nAvailable functions:")
# print("• predict_next_word(text, top_k=3) - Get multiple predictions")
# print("• predict_single_word(text) - Get best prediction only")
# print("• interactive_mode() - Interactive testing")

# print(f"\nModel Info:")
# vocab_size = len(tokenizer.word_index) + 1
# print(f"• Vocabulary: {vocab_size:,} words")
# print(f"• Input length: 5 words")
# print(f"• Model size: {model.count_params():,} parameters")

# print("\n💡 Examples:")
# print("predict_next_word('hello world how are')")
# print("predict_single_word('the cat is')")
# print("interactive_mode()")

# # Quick example
# print(f"\n🔥 Quick test:")
# example_text = "Most people check their emails "
# result = predict_single_word(example_text)
# print(f"Input: '{example_text}' → Prediction: '{result}'")
# ✅ Page config must be at the very top
import streamlit as st
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠")

# Now import the rest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import re

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('next_word_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ""
    text = text.lower()
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'= = .+ = =', '', text)
    text = re.sub(r'= .+ =', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Predict next words
def predict_next_word(input_text, top_k=3):
    cleaned_text = clean_text(input_text)
    tokens = tokenizer.texts_to_sequences([cleaned_text])[0]
    seq_length = 5
    if len(tokens) >= seq_length:
        input_sequence = tokens[-seq_length:]
    else:
        input_sequence = [0] * (seq_length - len(tokens)) + tokens
    input_sequence = np.array([input_sequence])
    predictions = model.predict(input_sequence, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if idx in tokenizer.index_word:
            word = tokenizer.index_word[idx]
            probability = predictions[idx]
            results.append((word, probability))
    return results

# Streamlit UI
st.title("🧠 Next Word Prediction")
st.write("Enter a sentence and get the next word predictions using your trained language model.")

# Input box
user_input = st.text_input("Enter your sentence:", "")

# Slider to select top-k predictions
top_k = st.slider("Number of predictions to show", min_value=1, max_value=10, value=3)

# Predict and show results
if user_input:
    try:
        predictions = predict_next_word(user_input, top_k=top_k)
        if predictions:
            st.subheader("Predicted Next Words:")
            for i, (word, prob) in enumerate(predictions, 1):
                st.write(f"**{i}.** `{word}` — {prob:.2%}")
        else:
            st.warning("No predictions available.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter a sentence above to get started.")

# Model summary info
with st.expander(" Model Details"):
    vocab_size = len(tokenizer.word_index) + 1
    st.write(f"**Vocabulary size:** {vocab_size:,} words")
    st.write(f"**Input sequence length:** 5 words")
    st.write(f"**Model parameters:** {model.count_params():,}")
