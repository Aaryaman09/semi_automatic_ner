import json, requests
from typing import Dict, List, Tuple
from icecream import ic


class PullDataFromCloud:
    def __init__(self, datasources:Dict, separators:Dict)->None:
        self.datasources = datasources
        self.separators= separators

    def _pull_data_util(self, source:str)->str:
        return requests.get(source).text.splitlines()

    def pull_data(self)->Tuple[List[str], List[str], List[str]]:
        return (
            self._pull_data_util(self.datasources.get('train')), 
            self._pull_data_util(self.datasources.get('test')), 
            self._pull_data_util(self.datasources.get('validation'))
        )
    
    def generate_tokens_and_tags(self, text_data:List[str], seperator:str)->Tuple[List[List[str]], List[List[str]]]:
        tokens = []
        tags = []

        temp_tokens = []
        temp_tags = []

        for line in text_data:
            if line != "":
                tag, token = line.strip().split(seperator)
                temp_tokens.append(tag)
                temp_tags.append(token)
            else:
                tokens.append(temp_tokens)
                tags.append(temp_tags)

                temp_tags, temp_tokens = [], []

        return tokens, tags
    
    def __call__(self)->Dict:
        train_str, test_str, validation_str = self.pull_data()

        train_tokens, train_tags = self.generate_tokens_and_tags(text_data=train_str, seperator=self.separators.get('train'))
        test_tokens, test_tags = self.generate_tokens_and_tags(text_data=test_str, seperator=self.separators.get('test'))
        validation_tokens, validation_tags = self.generate_tokens_and_tags(text_data=validation_str, seperator=self.separators.get('validation'))

        return {
            "train_data":{
                'tokens': train_tokens,
                'tags': train_tags
            },
            "test_data":{
                'tokens': test_tokens,
                'tags': test_tags
            },
            "validation_data":{
                'tokens': validation_tokens,
                'tags': validation_tags
            }
        }