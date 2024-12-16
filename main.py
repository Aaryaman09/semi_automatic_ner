from source_code.pull_data.pull_data_from_source import PullDataFromCloud 
from source_code.pull_data.dataset_prep import BERTModelDatasetBuilder
from source_code.token_preprocessing.alignemnt_processing import PreprocessingTokens
from source_code.trainer_script.model_trainer import ModelTrainer
from source_code.collation_and_metrics.metrics_script import MetricsForTokenClassififcation

import json, os
from icecream import ic

def read_json(path:os.path):
    with open(path, 'r') as file:
        config = json.load(file)
    
    return config

if __name__ == '__main__':
    # importing configs
    source_config = read_json(os.path.join('configs','source_data.json'))
    model_config = read_json(os.path.join('configs','model_config.json'))

    token_and_tags = PullDataFromCloud(
        datasources=source_config.get('source_config'),
        separators=source_config.get('separators'))()
    
    dataset, index2tag, tag2index = BERTModelDatasetBuilder(token_and_tags=token_and_tags)()

    preprocessing_token_obj = PreprocessingTokens(
        model_name=model_config.get('model_name'),
        dataset=dataset)

    metric_obj = MetricsForTokenClassififcation(tag2index=tag2index)

    tokenized_dataset = preprocessing_token_obj.adding_labels_and_input_ids()

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
    
    model_trainer.execute_training()

    model_trainer.save_model_locally()