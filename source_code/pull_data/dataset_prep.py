from datasets import Dataset, DatasetDict
import pandas as pd
from typing import Dict, List, Tuple
from icecream import ic

class BERTModelDatasetBuilder:
    def __init__(self, token_and_tags:Dict):
        self.token_and_tags = token_and_tags

    def bert_dataset_util(self, data:Dict)->Dataset:
        df = pd.DataFrame({"tokens":data.get('tokens'), "ner_tags_str":data.get('tags')})
        return Dataset.from_pandas(df)
    
    def generate_index2tag_and_tag2index(self, ner_tags_str:List[str])->Tuple[Dict, Dict]:
        unique_tags_set = set()

        for tag in ner_tags_str:
            unique_tags_set.update(tag)

        unique_tags = list(set([x[2:] for x in list(unique_tags_set) if x!='O']))

        tag2index = {"O":0}

        for tag in unique_tags:
            tag2index[f'B-{tag}'] = len(tag2index)
            tag2index[f'I-{tag}'] = len(tag2index)

        index2tag = {v:k for k,v in tag2index.items()}

        return index2tag, tag2index
    
    def attaching_tag2index_to_dataset(self, dataset:Dataset, tag2index:Dict)->Dataset:
        return dataset.map(lambda example: {'ner_tags':[tag2index[tag] for tag in example['ner_tags_str']]})
    
    def __call__(self)->Tuple[Dataset, Dict, Dict]:
        dataset = DatasetDict({
            'train':self.bert_dataset_util(data=self.token_and_tags.get('train_data')),
            'test':self.bert_dataset_util(data=self.token_and_tags.get('test_data')),
            'validation':self.bert_dataset_util(data=self.token_and_tags.get('validation_data'))
        })

        index2tag, tag2index = self.generate_index2tag_and_tag2index(ner_tags_str=dataset['train']['ner_tags_str'])

        dataset = self.attaching_tag2index_to_dataset(dataset=dataset,
                                                      tag2index=tag2index)
        
        return dataset, index2tag, tag2index
