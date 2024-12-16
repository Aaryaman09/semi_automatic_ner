from source_code.pull_data.pull_data_from_source import PullDataFromCloud 
from source_code.pull_data.dataset_prep import BERTModelDatasetBuilder

import json
from icecream import ic

if __name__ == '__main__':
    with open('configs\source_data.json', 'r') as file:
        config = json.load(file)

    token_and_tags = PullDataFromCloud(
        datasources=config.get('source_config'),
        separators=config.get('separators'))()
    
    dataset, index2tag, tag2index = BERTModelDatasetBuilder(token_and_tags=token_and_tags)()
