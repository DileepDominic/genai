import gradio as gr
import ollama

def predict(message, history):
    # Call your local Ollama model
    response = ollama.chat(model='qwen3.5:0.8b', messages=[
        {'role': 'user', 'content': message}
    ])
    return response['message']['content']

# Launch a simple chat interface
gr.ChatInterface(predict).launch()