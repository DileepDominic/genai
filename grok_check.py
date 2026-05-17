import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_SECRET_KEY = os.getenv("GROQ_API_KEY")

# 1. Connect to your LOCAL Ollama instance
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't check keys, but the SDK requires a string
)

# 2. Connect to your CLOUD Groq instance
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_SECRET_KEY
)

# Call Ollama (Local)
#local_response = ollama_client.chat.completions.create(
#    model="llama3",
#    messages=[{"role": "user", "content": "Hello local model!"}]
#)

# Call Groq (Cloud - Ultra Fast)
# Call Groq (Cloud - Using an active model)
cloud_response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # <--- Updated model name here
    messages=[{"role": "user", "content": "Hello super-fast cloud!"}]
)

print(cloud_response.choices[0].message.content)