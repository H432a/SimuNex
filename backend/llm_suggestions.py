from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import gc

load_dotenv()  


llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY"),  
    model="llama3-70b-8192"  
)

# Add at the top
import gc

def suggest_projects(components):
    try:
        prompt = f"Suggest 2-3 IoT projects using: {', '.join(components)}. Keep responses under 200 words."
        response = llm.invoke(prompt).content
        gc.collect()  # Clean up LLM memory
        return response
    except Exception as e:
        gc.collect()
        return f"Could not generate suggestions: {str(e)}"