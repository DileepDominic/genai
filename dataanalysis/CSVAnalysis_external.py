import gradio as gr
from openai import OpenAI
import pandas as pd
import os

from dotenv import load_dotenv

# -----------------------------------
# Load Environment Variables
# -----------------------------------
load_dotenv()

GROQ_SECRET_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_SECRET_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# -----------------------------------
# Global DataFrame Storage
# -----------------------------------
current_dataframe = None

# -----------------------------------
# Connect to Groq
# -----------------------------------
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_SECRET_KEY
)

# -----------------------------------
# CSV Upload Handler
# -----------------------------------
def load_csv_file(file):
    global current_dataframe

    if file is None:
        return None, "No file uploaded."

    try:
        # Read CSV safely
        df = pd.read_csv(
            file.name,
            encoding_errors="ignore"
        )

        current_dataframe = df

        # Summary
        summary = f"""
📊 File Loaded Successfully!

- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

### Columns
{', '.join(df.columns.tolist())}
"""

        return df.head(5), summary

    except Exception as e:
        return None, f"❌ Error loading CSV: {str(e)}"

# -----------------------------------
# Analyze Data Function
# -----------------------------------
def analyze_data(user_question, history):
    global current_dataframe

    if history is None:
        history = []

    # No CSV uploaded
    if current_dataframe is None:

        history.append(
            gr.ChatMessage(
                role="user",
                content=user_question
            )
        )

        history.append(
            gr.ChatMessage(
                role="assistant",
                content="Please upload a CSV file first."
            )
        )

        return history, ""

    try:
        # Dataset context
        csv_preview = current_dataframe.head(15).to_string()

        csv_columns = current_dataframe.dtypes.to_string()

        summary_stats = (
            current_dataframe
            .describe(include="all")
            .fillna("")
            .to_string()
        )

        missing_values = (
            current_dataframe
            .isnull()
            .sum()
            .to_string()
        )

        # Prompt
        engineered_prompt = f"""
You are a professional data analyst.

Analyze the dataset carefully.

[DATASET COLUMNS & TYPES]
{csv_columns}

[SUMMARY STATISTICS]
{summary_stats}

[MISSING VALUES]
{missing_values}

[FIRST 15 ROWS]
{csv_preview}

[USER QUESTION]
{user_question}

Rules:
- Answer only from available data.
- If data is insufficient, clearly mention it.
- Keep responses concise and factual.
"""

        # Groq API call
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": engineered_prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"⚠️ GROQ execution error: {str(e)}"

    # Append chat history
    history.append(
        gr.ChatMessage(
            role="user",
            content=user_question
        )
    )

    history.append(
        gr.ChatMessage(
            role="assistant",
            content=answer
        )
    )

    return history, ""

# -----------------------------------
# Build UI
# -----------------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📈 CSV Data Analyst Chatbot")

    gr.Markdown(
        """
Upload a CSV file and ask questions about your dataset using Groq LLM.
"""
    )

    with gr.Row():

        # Left Panel
        with gr.Column(scale=1):

            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"]
            )

            status_output = gr.Markdown(
                "No file uploaded yet."
            )

            data_preview = gr.DataFrame(
                label="CSV Preview (First 5 Rows)"
            )

        # Right Panel
        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="Analysis History",
                height=500
            )

            msg_input = gr.Textbox(
                label="Ask a question about your data",
                placeholder="e.g. summarize the dataset"
            )

            clear_btn = gr.Button("Clear Chat")

    # -----------------------------------
    # Upload Event
    # -----------------------------------
    file_input.change(
        fn=load_csv_file,
        inputs=file_input,
        outputs=[data_preview, status_output]
    )

    # -----------------------------------
    # Chat Submit Event
    # -----------------------------------
    msg_input.submit(
        fn=analyze_data,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    # -----------------------------------
    # Clear Chat
    # -----------------------------------
    clear_btn.click(
        lambda: [],
        None,
        chatbot,
        queue=False
    )

# -----------------------------------
# Launch App
# -----------------------------------
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft()
    )