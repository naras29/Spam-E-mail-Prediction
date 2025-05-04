import streamlit as st
import pickle
import re
import string

# Load the trained model and vectorizer
SVM = pickle.load(open('model.pkl', 'rb'))
tf_vector = pickle.load(open('vectorizer.pkl', 'rb'))


# Function to preprocess input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


# Streamlit app UI
st.title("Spam Email Classifier")
st.subheader("Check whether an E-mail message is spam or not.")

# User input
input_email = st.text_area("Enter the E-mail message below:")

if st.button("Predict"):
    if input_email.strip() == "":
        st.warning("Please enter some E-mail content.")
    else:
        # Preprocess and vectorize
        cleaned_input = clean_text(input_email)
        input_features = tf_vector.transform([cleaned_input])

        # Make prediction
        prediction = SVM.predict(input_features)[0]

        # Show result
        if prediction == "spam":
            st.error(" This is a SPAM message.")
        else:
            st.success("This is a HAM (non-spam) message.")


