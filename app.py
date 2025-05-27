import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

# Load the tokenizer, model and data

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./emotion_model")
    model = TFAutoModelForSequenceClassification.from_pretrained("./emotion_model")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

@st.cache_resource
def load_data():
    dataset = load_dataset("emotion", cache_dir="./cache")
    return dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = load_data()
# Separating the dataset into training, validation, and test sets
train_dataset = dataset["train"].map(tokenize_function, batched=True)

# Defining the Data Collator
data_collator = DataCollatorWithPadding(tokenizer=load_model_and_tokenizer()[0], return_tensors="tf",padding=True)

# Function to preprocess the dataset
def create_tf_dataset(dataset, tokenizer, batch_size=32, shuffle=True):
    # Convert to TensorFlow dataset with dynamic padding
    tf_dataset = dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        drop_remainder=False  # Keep all samples
    )
    return tf_dataset

# Creating tensorflow datasets
train_tf_dataset = create_tf_dataset(train_dataset, tokenizer, batch_size=32, shuffle= True)

# Testing with new sample input
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    predicted_class = int(predicted_class)
    return dataset["train"].features["label"].int2str(predicted_class)

# Streamlit app
st.title("Emotion Classification App")
st.write("Enter a sentence to classify its emotion:")
user = st.text_input("Input Sentence")
if st.button("Predict Emotion"):
    if user:
        emotion = predict_emotion(user)
        st.write(f"Predicted Emotion: {emotion}")
    else:
        st.write("Please enter a sentence to classify its emotion.")


# Sidebar with plot options
st.sidebar.title("Dataset Overview")
st.sidebar.write("This app uses the Emotion dataset for emotion classification.")
st.sidebar.write("Please choose an option to visualize the dataset:")
st_option = st.sidebar.selectbox(
    "Select an option",
    [
        "Emotion Label Distribution",
        "Emotion Text Length Distribution",
        "WordCloud",
        "Confusion Matrix",
    ]
)

emotion_labels = dataset["train"].features["label"].names
# Placeholder for future visualizations
if st_option == "Emotion Label Distribution":
    st.image("plots/emotion_label_distribution.png", caption="Emotion Label Distribution", use_column_width=True)
elif st_option == "Emotion Text Length Distribution":
    st.image("plots/emotion_text_length_distribution.png", caption="Emotion Text Length Distribution", use_column_width=True)
    st.image("plots/emotion_text_length_by_label.png", caption="Emotion Text Length by Label", use_column_width=True)
elif st_option == "WordCloud":
    for label in emotion_labels:
        plot=f"plots/wordcloud_{label.lower()}.png"
        st.image(plot, caption=f"Word Cloud for {label} Emotion", use_column_width=True)
elif st_option == "Confusion Matrix":
    st.image("plots/confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
else:
    st.write("Please select an option from the sidebar to visualize the dataset.")

