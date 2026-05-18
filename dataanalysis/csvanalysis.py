import gradio as gr
import ollama
import pandas as pd

# Global state to hold the data so we don't reload it every chat message
current_dataframe = None

def load_csv_file(file):
    global current_dataframe
    if file is None:
        return None, "No file uploaded."
    
    try:
        # Load file into Pandas
        df = pd.read_csv(file.name)
        current_dataframe = df
        
        # Create a text summary of the CSV schema/head for the user to see
        summary = f"📊 **File Loaded Successfully!**\n"
        summary += f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
        summary += f"- Columns found: `{', '.join(df.columns.tolist())}`"
        return df.head(5), summary
    except Exception as e:
        return None, f"❌ Error loading CSV: {str(e)}"

def analyze_data(user_question, history):
    global current_dataframe
    
    if current_dataframe is None:
        return history + [["User", user_question], ["Assistant", "Please upload a CSV file first before asking questions!"]]
    
    # 1. Gather context about the dataset
    # We send a clean preview of the dataframe and data types so the LLM understands it.
    csv_preview = current_dataframe.head(15).to_string() 
    csv_columns = current_dataframe.dtypes.to_string()
    
    # 2. Build a data-focused prompt wrapper
    engineered_prompt = f"""
You are a data analyst assistant. Analyze the CSV dataset provided below to answer the user's question.

[DATASET ARCHITECTURE & COLUMNS]
{csv_columns}

[FIRST 15 ROWS SAMPLE DATA]
{csv_preview}

[USER QUESTION]
{user_question}

Provide a concise, accurate, data-driven answer based ONLY on the data structure and data sample visible above.
"""

    try:
        # 3. Request inference from local Qwen
        # (Remember to switch to 'qwen3.5:0.8b' if your computer is still running too hot!)
        response = ollama.chat(
            model='qwen3.5:0.8b', 
            messages=[{'role': 'user', 'content': engineered_prompt}]
        )
        answer = response['message']['content']
    except Exception as e:
        answer = f"⚠️ Ollama execution error: {str(e)}"
    
    # Append conversation to the chat UI list format
    history.append((user_question, answer))
    return history, "" # Clear the input textbox

# --- Building the Gradio Custom Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 Local CSV Data Analyst Chatbot")
    gr.Markdown("Upload a CSV file and ask your local Qwen model to explain, query, or analyze it.")
    
    with gr.Row():
        # Left Panel: File uploads and Previewers
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            status_output = gr.Markdown("No file uploaded yet.")
            data_preview = gr.DataFrame(label="CSV Data Preview (First 5 Rows)")
            
        # Right Panel: Chat Framework
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Analysis History")
            msg_input = gr.Textbox(label="Ask a question about your data", placeholder="e.g., What columns are here? Summarize this data...")
            clear_btn = gr.Button("Clear Chat")

    # --- Wire up UI Logic components ---
    
    # When a file is uploaded, parse it and update the preview and markdown status
    file_input.change(
        fn=load_csv_file, 
        inputs=[file_input], 
        outputs=[data_preview, status_output]
    )
    
    # When user hits enter or submits text, fire the analysis
    msg_input.submit(
        fn=analyze_data, 
        inputs=[msg_input, chatbot], 
        outputs=[chatbot, msg_input]
    )
    
    # Clear history button function
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# Launch the custom UI app
if __name__ == "__main__":
    demo.launch()