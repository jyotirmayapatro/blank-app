import streamlit as st
import openai
import os

# Set the title for the Streamlit app
st.title("JP's Super Awesome App for Travel Experience!")

# Input field for user travel experience
user_prompt = st.text_area("What is your latest travel experience?")

# Load the OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# OpenAI function to generate a response
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant for travel experiences."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Generate and display the response based on user input
if st.button("Submit Feedback"):
    if user_prompt:
        response = generate_response(user_prompt)
        st.write(response)
    else:
        st.write("Please enter your experience.")
