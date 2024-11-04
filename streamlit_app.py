import streamlit as st
from openai import OpenAI
import os

# JP's application title
st.title("JP's Super Awesome App for Travel Experience!")

# User input for travel experience
user_prompt = st.text_area("What is your latest travel experience?")

# Load your API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI"]["OpenAIkey"]

# Initiate OpenAI
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3 Types of Experiences:
from langchain.prompts import PromptTemplate

# Handling Negative Experiences Caused by the Airline:
negative_airline_template = PromptTemplate(
    template="""If the text below describes a negative experience due to an airline issue, 
    generate a response offering sympathies and mentioning customer service will follow up. 
    Otherwise, say nothing.\n\nText: {text}""",
    input_variables=["text"]
)

# Handling Negative Experiences Beyond the Airline's Control:
negative_external_template = PromptTemplate(
    template="""If the text below describes a negative experience not due to an airline (e.g., weather-related delay),
    respond with a sympathetic message, but explain that the airline is not liable.\n\nText: {text}""",
    input_variables=["text"]
)

# Handling Positive Experiences:
positive_experience_template = PromptTemplate(
    template="""If the text below describes a positive experience, respond with a thank-you message.\n\nText: {text}""",
    input_variables=["text"]
)

# Creating chains for each template:
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

negative_caused_by_the_airline_chain = LLMChain(llm=llm, prompt=negative_airline_template, output_parser=StrOutputParser())
negative_beyond_airline_control_chain = LLMChain(llm=llm, prompt=negative_external_template, output_parser=StrOutputParser())
positive_experience_chain = LLMChain(llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser())

# Based on user's input, the app will provide a response:
if st.button("Submit Feedback"):
    if user_prompt:  # Use user_prompt consistently
        # if the feedback is caused by the Airline
        response = negative_caused_by_the_airline_chain.run({"text": user_prompt}).strip()
        
        if response:
            st.write(response)
        else:
            # if the feedback is beyond the Airline's Control
            response = negative_beyond_airline_control_chain.run({"text": user_prompt}).strip()
            
            if response:
                st.write(response)
            else:
                # if the feedback is Positive Experience
                response = positive_experience_chain.run({"text": user_prompt}).strip()
                
                if response:
                    st.write(response)
                else:
                    # Default message if no specific response
                    st.write("Thank you for your feedback.")
    else:
        st.write("Please enter your experience.")
