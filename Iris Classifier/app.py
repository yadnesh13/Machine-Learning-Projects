import streamlit as st
from classifier import predict_labels

def main():
    st.title("Iris Flower Classifier")

    # Add input fields for features
    sepal_length = st.slider("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.slider("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.slider("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.slider("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

    # Create feature vector from user input
    input_features = [sepal_length, sepal_width, petal_length, petal_width]

    # Make prediction using the classifier
    predicted_label = predict_labels(input_features)

    # Display predicted label
    st.write(f"Predicted Iris Species: {predicted_label}")

if __name__ == "__main__":
    main()
