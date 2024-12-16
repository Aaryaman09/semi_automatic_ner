from transformers import pipeline
from icecream import ic
from typing import Dict

class Predictor:
    def __init__(self, prediction_config:Dict):
        self.pipe = pipeline(**prediction_config)

    def predict(self, input_str:str):
        return self.pipe(input_str)
    
if __name__=='__main__':
    predictor_obj = Predictor(
        task="token-classification",
        model="ner_distilbert",
        aggregation_strategy="simple")
    
    ic(predictor_obj.predict("which restaurant serves the best shushi in new york?"))
