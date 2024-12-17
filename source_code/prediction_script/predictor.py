from transformers import pipeline
from typing import Dict
from source_code.util import read_json
import os

class Predictor:
    def __init__(self, prediction_config:Dict):
        self.pipe = pipeline(**prediction_config)

    def predict(self, input_str:str):
        return self.pipe(input_str)


#####################################################################################
#####################################################################################


# Gradio NER prediction function
def ner_predict(text:str):
    """Perform NER prediction on input text."""
    prediction_config = read_json(path = os.path.join('configs','prediction_config.json')).get('prediction_config')
    predictor_obj = Predictor(prediction_config=prediction_config)
    entities = predictor_obj.predict(text) ## Predictions
    return {"text": text, "entities": entities}
