from datasets import Dataset, DatasetDict
import pandas as pd
from typing import Dict

class BERTModelDatasetBuilder:
    def __init__(self, token_and_tags:Dict):
        self.token_and_tags = token_and_tags

    def bert_dataset_util(self, data:Dict):
        df = pd.DataFrame({"tokens":data.get('tokens'), "ner_tags_str":data.get('tags')})
        return Dataset.from_pandas(df)
    
    def __call__(self):
        return DatasetDict({
            'train':self.bert_dataset_util(data=self.token_and_tags.get('train_data')),
            'test':self.bert_dataset_util(data=self.token_and_tags.get('test_data')),
            'validation':self.bert_dataset_util(data=self.token_and_tags.get('validation_data'))
        })
