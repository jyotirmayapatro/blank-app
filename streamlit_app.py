import streamlit as st
import openai
import os

# JP's application title
st.title("JP's Super Awesome App for Travel Experience!")

# User input for travel experience
user_prompt = st.text_area("What is your latest travel experience?")

# Load your API Key
api_key = st.secrets.get("OpenAIkey") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is not set. Please configure it in Streamlit secrets or environment variables.")
openai.api_key = api_key

# Define templates for different types of responses
from langchain.prompts import PromptTemplate

# Original template names
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

# Create response chains for each template using original chain names
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI model
llm = ChatOpenAI(api_key=api_key, model="gpt-4")

# Original chain names
negative_caused_by_the_airline_chain = LLMChain(
    llm=llm, prompt=negative_caused_by_the_airline_template, output_parser=StrOutputParser()
)
negative_beyond_airline_control_chain = LLMChain(
    llm=llm, prompt=negative_beyond_airline_control_template, output_parser=StrOutputParser()
)
positive_experience_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)

# Based on user's input, provide a response
if st.button("Submit Feedback"):
    if user_prompt:
        # Handle feedback caused by the airline
        response = negative_caused_by_the_airline_chain.run({"text": user_prompt}).strip()

        if response:
            st.write(response)
        else:
            # Handle feedback beyond the airline's control
            response = negative_beyond_airline_control_chain.run({"text": user_prompt}).strip()

            if response:
                st.write(response)
            else:
                # Handle positive feedback
                response = positive_experience_chain.run({"text": user_prompt}).strip()

                if response:
                    st.write(response)
                else:
                    # Default message if no specific response
                    st.write("Thank you for your feedback.")
    else:
        st.write("Please enter your experience.")
