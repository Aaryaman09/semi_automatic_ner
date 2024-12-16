from transformers import AutoTokenizer
from datasets import Dataset

class PreprocessingTokens:
    def __init__(self, model_name:str, dataset:Dataset):
        self.model_name= model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation = True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # if id=-100 then loss is not calculated
                if word_idx is None:
                    label_ids.append(-100)
                
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])

                else:
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels

        return tokenized_inputs
    
    def adding_labels_and_input_ids(self)->Dataset:
        return self.dataset.map(self.tokenize_and_align_labels, batched=True)
