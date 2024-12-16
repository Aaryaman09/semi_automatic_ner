from transformers import (AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          Trainer, 
                          TrainingArguments)

from typing import Dict
from datasets import Dataset
import os

class ModelTrainer:
    def __init__(
            self, 
            model_name:str, 
            tokenizer, 
            tokenized_dataset:Dataset, 
            compute_metrics, 
            index2tag:Dict, 
            tag2index:Dict, 
            training_arguments:Dict, 
            trained_model_loc:os.path):
        
        self.trained_model_loc = trained_model_loc
        self.trainer = Trainer(
            model=AutoModelForTokenClassification.from_pretrained(model_name, id2label=index2tag, label2id=tag2index),
            args=TrainingArguments(**training_arguments),
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            tokenizer=tokenizer)
        
    def execute_training(self)->None:
        self.trainer.train()

    def save_model_locally(self):
        self.trainer.save_model(self.trained_model_loc)