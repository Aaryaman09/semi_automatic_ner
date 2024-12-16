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
    
    def generate_tokens_and_tags(self, text_data:List[str], seperator:str):
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
    
    def pull_train_test_data(self)->Dict:
        train_str, test_str, validation_str = self.pull_data()

        train_tokens, train_tags = self.generate_tokens_and_tags(text_data=train_str, seperator=self.separators.get('train'))
        test_tokens, test_tags = self.generate_tokens_and_tags(text_data=test_str, seperator=self.separators.get('test'))
        validation_tokens, validation_tags = self.generate_tokens_and_tags(text_data=validation_str, seperator=self.separators.get('validation'))

        return {
            "train_data":{
                'tokens': train_tokens[0],
                'tags': train_tags[0]
            },
            "test_data":{
                'tokens': test_tokens[0],
                'tags': test_tags[0]
            },
            "validation_data":{
                'tokens': validation_tokens[0],
                'tags': validation_tags[0]
            }
        }


if __name__ == '__main__':
    with open('configs\source_data.json', 'r') as file:
        config = json.load(file)

    ic(PullDataFromCloud(datasources=config.get('source_config'),
                      separators=config.get('separators')).pull_train_test_data())
        