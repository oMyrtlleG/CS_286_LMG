import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- HELPER 1: UTILITY FUNCTION FOR GEMINI API CALL ---
def _call_gemini_api(messages, model_name, api_key, max_tokens=5000):
    """
    Makes a call to the Gemini API using the google-genai SDK.
    """
    try:
        # Initialize the client with the key
        client = genai.Client(api_key=api_key)

        # The first item in 'messages' is the System Instruction
        system_instruction = messages[0]['content']
        user_message = messages[1]['content']

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            temperature=0.3,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=config,
        )

        if response.candidates and response.candidates[0].content.parts:
            return response.text.strip()
        else:
            return "Generation failed: Empty or blocked response from Gemini API."

    except APIError as e:
        st.error(f"Gemini API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during the Gemini API call: {e}")
        return None