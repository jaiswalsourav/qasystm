import os
from venv import logger
from dotenv import load_dotenv
import sys
from llama_index.llms.google_genai import GoogleGenAI
# Assuming these are your local files
# from exception import customexception 
# from logger import logging

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info("GOOGLE_API_KEY loaded successfully")
logger.info(GOOGLE_API_KEY)
def load_model():
    """
    Loads a Gemini Flash model using the modern LlamaIndex GoogleGenAI class.
    
    Returns:
    - GoogleGenAI: Initialized instance for Gemini 2.5 Flash.
    """
    try:
        # 1. Parameter is 'model', not 'models'
        # 2. Use the current stable 2026 model string
        model = GoogleGenAI(
            model="models/gemini-2.5-flash", 
            api_key=GOOGLE_API_KEY
        )
        return model
    except Exception as e:
        # Ensure customexception is properly imported or replaced with standard Exception
        raise Exception(f"Failed to load Gemini model: {e}")
        
        
