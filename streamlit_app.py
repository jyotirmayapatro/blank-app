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

# Define templates for each condition
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
from langchain_core.runnables import RunnableBranch

# Initialize OpenAI model
llm = ChatOpenAI(api_key=openai.api_key, model="gpt-3.5-turbo")

# Chains for each type of response
negative_caused_by_the_airline_chain = LLMChain(
    llm=llm, prompt=negative_caused_by_the_airline_template, output_parser=StrOutputParser()
)

negative_beyond_airline_control_chain = LLMChain(
    llm=llm, prompt=negative_beyond_airline_control_template, output_parser=StrOutputParser()
)

positive_experience_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)

# Set up RunnableBranch for conditional routing
branch = RunnableBranch(
    (lambda x: "bad" in x["text"].lower() and "airline" in x["text"].lower(), negative_caused_by_the_airline_chain),
    (lambda x: "bad" in x["text"].lower() and "airline" not in x["text"].lower(), negative_beyond_airline_control_chain),
    positive_experience_chain  # Default if no conditions are met
)

# Use RunnableBranch to invoke the appropriate chain based on the input
if st.button("Submit Feedback"):
    if user_prompt:
        # Invoke branch and handle dictionary output
        response = branch.invoke({"text": user_prompt})
        response_content = response.get("text", "").strip()  # Extract content if available
        st.write(response_content if response_content else "Thank you for your feedback.")
    else:
        st.write("Please enter your experience.")
