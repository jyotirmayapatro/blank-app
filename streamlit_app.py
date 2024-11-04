import streamlit as st
import openai
import os

# JP's application title
st.title("JP's Super Awesome App for Travel Experience!")

# User input for travel experience
user_prompt = st.text_area("What is your latest travel experience?")

# Load OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Define different types of responses (3)
from langchain.prompts import PromptTemplate

negative_caused_by_the_airline_template = PromptTemplate(
    template="""If the text below describes a negative experience due to an airline issue, 
    generate a response offering sympathies and mentioning customer service will follow up. 
    Otherwise, say nothing.\n\nText: {text}""",
    input_variables=["text"]
)

negative_beyond_airline_control_template = PromptTemplate(
    template="""If the text below describes a negative experience not due to an airline (e.g., weather-related delay),
    respond with a sympathetic message, but explain that the airline is not liable.\n\nText: {text}""",
    input_variables=["text"]
)

positive_experience_template = PromptTemplate(
    template="""If the text below describes a positive experience, respond with a thank-you message.\n\nText: {text}""",
    input_variables=["text"]
)

from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

# OpenAI model
llm = ChatOpenAI(api_key=openai.api_key, model="gpt-3.5-turbo")

# Chains
negative_caused_by_the_airline_chain = LLMChain(
    llm=llm, prompt=negative_caused_by_the_airline_template, output_parser=StrOutputParser()
)

negative_beyond_airline_control_chain = LLMChain(
    llm=llm, prompt=negative_beyond_airline_control_template, output_parser=StrOutputParser()
)

positive_experience_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)

# Import RunnableBranch and set up branching logic
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "bad" in x["text"].lower() and "airline" in x["text"].lower(), negative_caused_by_the_airline_chain),
    (lambda x: "bad" in x["text"].lower() and "airline" not in x["text"].lower(), negative_beyond_airline_control_chain),
    positive_experience_chain  # Default to this chain if neither condition matches
)

# Use branch to process user input and give a response
if st.button("Submit Feedback"):
    if user_prompt:
        try:
            # Call the branch directly with the input data
            response = branch({"text": user_prompt})
            
            # Check if the response is a string and display it
            if isinstance(response, str):
                st.write(response.strip())
            else:
                st.write("Received a non-string response:", response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter your experience.")
