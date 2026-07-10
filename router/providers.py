import os
from dotenv import load_dotenv
from groq import Groq
from google import genai as google_genai

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_small(query: str) -> dict:
    """Small tier: Groq llama-3.1-8b-instant"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": query}],
    )
    return {
        "text": response.choices[0].message.content,
        "tier": "small",
        "model": "llama-3.1-8b-instant",
        "provider": "groq",
    }

def call_mid(query: str) -> dict:
    """Mid tier: Groq llama-3.3-70b-versatile"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": query}],
    )
    return {
        "text": response.choices[0].message.content,
        "tier": "mid",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
    }

def call_frontier(query: str) -> dict:
    """Frontier tier: Gemini Flash Latest"""
    response = gemini_client.models.generate_content(
        model="gemini-flash-latest",
        contents=query,
    )
    return {
        "text": response.text,
        "tier": "frontier",
        "model": "gemini-flash-latest",
        "provider": "gemini",
    }
