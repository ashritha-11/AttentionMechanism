
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention # Make sure to import this if it's used in your custom model
import numpy as np
import pickle
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration --- #
MAX_LENGTH = 500 # Should match the MAX_LENGTH used during training

# --- Helper Functions --- #
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_reading_time(text):
    words = text.split()
    wpm = 200 # Average reading speed in words per minute
    minutes = len(words) / wpm
    return round(minutes, 2)

# --- Load Model Components --- #
@st.cache_resource
def load_model_components():
    # Load the Keras model
    model = tf.keras.models.load_model('self_attention_model.h5', custom_objects={'MultiHeadAttention': MultiHeadAttention})

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the label encoder (mapping dictionary)
    with open('label_encoder.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Create an attention extraction model
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, MultiHeadAttention):
            attention_layer = layer
            break
    if attention_layer is None:
        st.error("MultiHeadAttention layer not found in the model. Cannot extract attention scores.")
        return model, tokenizer, reverse_label_mapping, None

    embedding_output_for_attention = model.layers[1].output # Assuming embedding layer is the second
    _, attention_scores_extracted = attention_layer(
        embedding_output_for_attention,
        embedding_output_for_attention,
        return_attention_scores=True
    )
    attention_model = Model(inputs=model.input, outputs=attention_scores_extracted)

    return model, tokenizer, reverse_label_mapping, attention_model

model, tokenizer, reverse_label_mapping, attention_model = load_model_components()

# --- Streamlit App --- #
st.set_page_config(page_title="AI News Intelligence System", layout="wide")
st.title("AI News Intelligence System")

# Section 1: Enter News Article
st.header("Enter News Article")
news_article = st.text_area("Paste news article here...", height=250)

# Section 2: Predict Category
st.header("Predict Category")
if st.button("Analyze Article"):
    if news_article:
        # Preprocess the article
        clean_article = clean_text(news_article)
        sequence = tokenizer.texts_to_sequences([clean_article])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")

        # Make prediction
        prediction = model.predict(padded_sequence, verbose=0)[0]
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[predicted_class_index] * 100
        predicted_category = reverse_label_mapping.get(predicted_class_index, "Unknown")

        # Section 3: Predicted Category
        st.subheader("Predicted Category")
        st.success(f"Category: **{predicted_category}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # --- Attention Score Extraction for Important Words and Heatmap ---
        if attention_model is not None:
            attention_weights = attention_model.predict(padded_sequence, verbose=0)
            average_attention_weights = np.mean(attention_weights[0], axis=0)

            word_index = tokenizer.word_index
            reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

            actual_words = []
            for token_id in padded_sequence[0]:
                if token_id != 0: # 0 is typically for padding
                    actual_words.append(reverse_word_index.get(token_id, '<OOV>'))

            sentence_length = len(actual_words)
            if sentence_length > 0: # Ensure there are actual words to process
                received_attention_scores = np.sum(average_attention_weights[:sentence_length, :sentence_length], axis=0)
                word_attention_summary = dict(zip(actual_words, received_attention_scores))
                sorted_word_attention = sorted(word_attention_summary.items(), key=lambda item: item[1], reverse=True)

                # Section 4: Important Words (Simplified attention visualization)
                st.subheader("Important Words")
                # Display top N important words
                num_important_words = min(10, len(sorted_word_attention))
                for i in range(num_important_words):
                    word, score = sorted_word_attention[i]
                    st.markdown(f"**{word}**: {score:.4f} ")

                # Section 5: Attention Heatmap
                st.subheader("Attention Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    average_attention_weights[:sentence_length, :sentence_length],
                    cmap='viridis',
                    annot=False, # Set to False for cleaner heatmap with many words
                    fmt=".2f",
                    xticklabels=actual_words,
                    yticklabels=actual_words,
                    ax=ax
                )
                ax.set_title("Attention Heatmap")
                ax.set_xlabel("Keys (Words Attended To)")
                ax.set_ylabel("Queries (Attending Words)")
                st.pyplot(fig)
            else:
                st.warning("No meaningful words found for attention analysis.")

        # Section 6: Article Statistics
        st.subheader("Article Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", len(news_article.split()))
        with col2:
            st.metric("Character Count", len(news_article))
        with col3:
            st.metric("Reading Time (min)", calculate_reading_time(news_article))
    else:
        st.warning("Please enter a news article to analyze.")

