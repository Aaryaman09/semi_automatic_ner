import gradio as gr
from source_code.prediction_script.predictor import ner_predict
from source_code.util import read_json, save_config

import threading
import time, os

## -- IMPORTANT -- ## 
################################################################################################
# To have running logs in UI. I have kept trainer code in gradio app. Log as global variable. 
# If running logging is not required, this can be easily sent to source code folder, making code
# more modular. 
################################################################################################

from source_code.pull_data.pull_data_from_source import PullDataFromCloud 
from source_code.pull_data.dataset_prep import BERTModelDatasetBuilder
from source_code.token_preprocessing.alignemnt_processing import PreprocessingTokens
from source_code.trainer_script.model_trainer import ModelTrainer
from source_code.collation_and_metrics.metrics_script import MetricsForTokenClassififcation
from source_code.util import read_json, time_right_now

import os

# Function to start training
training_logs = []
training_status = "Not started"
def train_ner_model():
    """Training of the NER model on an IOB file."""
    global training_logs, training_status
    try:
        # Reset logs and status
        training_logs = []
        training_status = "Initializing..."

        # importing configs
        training_logs.append("fetching configs successfully.")
        training_status = "Running"
        source_config = read_json(os.path.join('configs','source_data.json'))
        model_config = read_json(os.path.join('configs','model_config.json'))
        training_logs.append("Configs fetched successfully.")

        # Pull Data from data sources
        training_logs.append("Fetching dataset.")
        token_and_tags = PullDataFromCloud(
            datasources=source_config.get('source_config'),
            separators=source_config.get('separators'))()
        training_logs.append("Dataset fetched successfully.")

        # Generating index2tag and tag2index
        training_logs.append("Generating index2tag and tag2index.")
        dataset, index2tag, tag2index = BERTModelDatasetBuilder(token_and_tags=token_and_tags)()
        training_logs.append("index2tag and tag2index generated successfully.")

        # Creating a tokenized Dataset for DistilBERT model to train, validate on.
        training_logs.append("Creating a tokenized Dataset for DistilBERT model to train, validate on.")
        preprocessing_token_obj = PreprocessingTokens(
            model_name=model_config.get('model_name'),
            dataset=dataset)
        
        tokenized_dataset = preprocessing_token_obj.adding_labels_and_input_ids()
        training_logs.append("Tokenized Dataset for DistilBERT model created successfully.")


        # Configuring trainer for DistilBERT model.
        training_logs.append("Configuring trainer for DistilBERT model.")
        metric_obj = MetricsForTokenClassififcation(tag2index=tag2index)

        model_trainer = ModelTrainer(
            model_name=model_config.get('model_name'),
            tokenizer=preprocessing_token_obj.tokenizer,
            tokenized_dataset=tokenized_dataset,
            compute_metrics=metric_obj.compute_metrics,
            index2tag=index2tag,
            tag2index=tag2index,
            training_arguments=model_config.get('training_arguments'),
            trained_model_loc=model_config.get('fine_tuned_model_dir')
            )
        training_logs.append("Trainer configured for DistilBERT model.")

        start_current_time_utc, one_hour_later = time_right_now()
        training_logs.append(f"Initializing model training. Est. time 45 mins to 60 mins. Current time : {start_current_time_utc}. Please check at time : {one_hour_later}")
        model_trainer.execute_training()
        end_current_time_utc, _ = time_right_now()
        # Calculate time taken

        time_taken = end_current_time_utc - start_current_time_utc

        # Extract hours, minutes, and seconds
        hours, remainder = divmod(time_taken.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        training_logs.append(f"Trainging Successful. Time Taken: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        training_status = "Completed"

        training_logs.append("Saving model.")
        training_status = "Saving Model"
        model_trainer.save_model_locally()
        training_logs.append("Model saved.")
        training_status = "Model Saved"

    except Exception as e:
        training_logs.append(f"Error during training: {str(e)}")
        training_status = "Failed"

###############################################################################################
###############################################################################################


# Markdown with an image and hyperlink
github_repo_link = "https://github.com/Aaryaman09/semi_automatic_ner"
github_markdown_content = f"""
<a href="{github_repo_link}" target="_blank" style="display: flex; align-items: center;">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
         alt="GitHub Logo" width="40" height="40" style="margin-right: 10px;">
    <span>Click image to access the GitHub repo and dataset</span>
</a>
"""

prediction_example = """
I recently visited The Green Bistro in downtown New York, and I had the most delicious pasta with fresh basil. 
The service was impeccable, and I highly recommend their signature cocktails. 
My friend had a vegetarian pizza, and we enjoyed a bottle of Italian red wine.
"""
###############################################################################################
## APPLICATION ##
###############################################################################################


# Gradio interface for prediction and dataset upload
with gr.Blocks() as ner_ui:
    gr.Markdown("# Named Entity Recognition (NER) Model UI on restaurant dataset")
    with gr.Row():
        gr.HTML(github_markdown_content)

    with gr.Tab("Prediction"):
        gr.Markdown("### Enter text for NER prediction")
        text_input = gr.Textbox(label="Input Text", placeholder="Enter text here...", value=prediction_example)
        prediction_output = gr.JSON(label="Predicted Entities")
        predict_button = gr.Button("Predict")
        predict_button.click(ner_predict, inputs=text_input, outputs=prediction_output)

    with gr.Tab("Configurations and Model Training"):
        gr.Markdown("### Configuration Files")

        gr.JSON(value=read_json(os.path.join('configs','source_data.json')), label="Source Config")
        with gr.Row():
            gr.JSON(value=read_json(os.path.join('configs','model_config.json')), label="Model Config")
            gr.JSON(value=read_json(os.path.join('configs','prediction_config.json')), label="Prediction Config")

        gr.Markdown("### As gr.JSON does not allow, please update the necessary config in bottom text box.###")

        with gr.Accordion("Source Configurations", open=False):  
            source_config_editor = gr.Textbox(value=read_json(os.path.join('configs','source_data.json')), label="Source Config", lines=10)
        with gr.Accordion("Model Configurations", open=False):
            model_config_editor = gr.Textbox(value=read_json(os.path.join('configs','model_config.json')), label="Model Config", lines=10)
        with gr.Accordion("Prediction Configurations", open=False):
            prediction_config_editor = gr.Textbox(value=read_json(os.path.join('configs','prediction_config.json')), label="Prediction Config", lines=10)

        save_button = gr.Button("Save Configurations")
        save_status = gr.Label()

        save_button.click(
            save_config, 
            inputs=[source_config_editor, model_config_editor, prediction_config_editor], 
            outputs=save_status
        )

        gr.Markdown("### Train the NER model on a custom dataset")
        finalize_config_checkbox = gr.Checkbox(label="Finalize Configuration", value=False)
        training_logs_output = gr.Textbox(label="Training Logs", lines=10, interactive=False)
        training_status_output = gr.Label(label="Training Status")
        train_button = gr.Button("Start Training", interactive=False)

        def toggle_train_button(is_finalized):
            """Enable or disable the training button based on config finalization."""
            return gr.update(interactive=is_finalized)

        finalize_config_checkbox.change(
            toggle_train_button, 
            inputs=finalize_config_checkbox, 
            outputs=train_button
        )

        def start_training():
            global training_logs, training_status
            # Start the training in a separate thread
            thread = threading.Thread(target=train_ner_model)
            thread.start()
            while thread.is_alive():
                time.sleep(1)
                yield "\n".join(training_logs), training_status
            yield "\n".join(training_logs), training_status

        train_button.click(start_training, outputs=[training_logs_output, training_status_output])

# Launch the Gradio interface
ner_ui.launch()
