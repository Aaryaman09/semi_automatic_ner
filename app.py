import gradio as gr
import pandas as pd
import requests
from io import StringIO
from icecream import ic

import json, os
def read_json(path:os.path):
    with open(path, 'r') as file:
        config = json.load(file)
    
    return config

from source_code.prediction_script.predictor import Predictor

# Dummy NER prediction function
def ner_predict(text:str):
    """Perform NER prediction on input text."""
    prediction_config = read_json(path = os.path.join('configs','prediction_config.json')).get('prediction_config')
    predictor_obj = Predictor(prediction_config=prediction_config)
    
    entities = predictor_obj.predict(text) ## Predictions

    return {"text": text, "entities": entities}

# Function to load dataset from file or GitHub link
def load_dataset(uploaded_file=None, github_link=None):
    """Load a dataset from an uploaded file or GitHub link."""
    try:
        if uploaded_file is not None:
            # Handle file upload
            if uploaded_file.name.endswith(".txt"):
                data = uploaded_file.read().decode("utf-8")
                return data[:500]  # Return a sample of the data
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                return df.head().to_csv(index=False)
            else:
                return "Unsupported file type. Please upload a .txt or .csv file."

        elif github_link:
            # Handle GitHub raw link
            response = requests.get(github_link)
            if response.status_code == 200:
                content_type = response.headers['Content-Type']
                if 'text/plain' in content_type:
                    return response.text[:500]  # Return a sample of the text
                elif 'text/csv' in content_type:
                    df = pd.read_csv(StringIO(response.text))
                    return df.head().to_csv(index=False)
                else:
                    return "Unsupported content type from the link."
            else:
                return f"Failed to fetch the file. HTTP Status: {response.status_code}"

        else:
            return "Please upload a file or provide a GitHub raw link."

    except Exception as e:
        return f"Error loading dataset: {str(e)}"

# Gradio interface for prediction and dataset upload
with gr.Blocks() as ner_ui:
    gr.Markdown("# Named Entity Recognition (NER) Model UI")

    with gr.Tab("Prediction"):
        gr.Markdown("### Enter text for NER prediction")
        text_input = gr.Textbox(label="Input Text", placeholder="Enter text here...")
        prediction_output = gr.JSON(label="Predicted Entities")
        predict_button = gr.Button("Predict")
        predict_button.click(ner_predict, inputs=text_input, outputs=prediction_output)

    with gr.Tab("Upload Dataset"):
        gr.Markdown("### Upload your dataset or provide a GitHub raw link")
        file_input = gr.File(label="Upload File (.txt or .csv)")
        github_input = gr.Textbox(label="GitHub Raw Link", placeholder="Enter GitHub raw link here...")
        dataset_output = gr.Textbox(label="Dataset Preview (First 500 characters or rows)", lines=10)
        load_button = gr.Button("Load Dataset")
        load_button.click(load_dataset, inputs=[file_input, github_input], outputs=dataset_output)

# Launch the Gradio interface
ner_ui.launch()
