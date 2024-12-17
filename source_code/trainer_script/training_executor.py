from source_code.pull_data.pull_data_from_source import PullDataFromCloud 
from source_code.pull_data.dataset_prep import BERTModelDatasetBuilder
from source_code.token_preprocessing.alignemnt_processing import PreprocessingTokens
from source_code.trainer_script.model_trainer import ModelTrainer
from source_code.collation_and_metrics.metrics_script import MetricsForTokenClassififcation
from source_code.util import read_json, time_right_now

import os

# Function to start training - Same code kept in gradio app - This is where we can place it to increase code modularity.
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