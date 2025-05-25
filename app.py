import streamlit as st
import pickle

model= pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Hate Speech Detection App")
st.write("Enter a sentence to check if it contains hate speech:")
user_input = st.text_input("")

if st.button("Predict"):
    if user_input:
        # Transform the input text using the vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # Make prediction using the loaded model
        prediction = model.predict(input_vector)
        
        # Display the result
        if prediction == 0:
            st.write("Hate Speech Detected: The input text contains hateful or threatening language")
        elif prediction== 1:
            st.write("Offensive Language Detected: The input text  May include profane, vulgar, or offensive language")
        elif prediction == 2:
            st.write("The input text is neutral.")
    else:
        st.warning("Please enter some text to analyze.")