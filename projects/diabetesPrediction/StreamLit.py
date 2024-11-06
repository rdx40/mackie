import numpy as np
import pickle
import streamlit as st

# Load the model with error handling
try:
    loaded_model = pickle.load(open('/home/omar/Downloads/trained_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

def diabetes_prediction(input_data):
    try:
        # Convert input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)

        # Reshape the array for a single instance prediction
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make a prediction
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            return 'The person is not diabetic'
        else:
            return 'The person is diabetic'
    except ValueError as e:
        return f"Invalid input data: {e}"

def main():
    # Set title
    st.title('Diabetes Prediction Web App')

    # Collect input from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Initialize prediction result
    diagnosis = ''

    # Prediction button
    if st.button('Diabetes Test Result'):
        try:
            # Ensure all inputs are provided and convert to float
            diagnosis = diabetes_prediction([
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ])
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

    # Display the result
    st.success(diagnosis)

if __name__ == '__main__':
    main()
