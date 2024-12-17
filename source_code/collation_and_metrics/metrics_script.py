import evaluate
import numpy as np
from typing import Dict

class MetricsForTokenClassififcation:
    def __init__(self, tag2index:Dict):
        self.metric = evaluate.load('seqeval')
        self.label_names = list(tag2index)

    def compute_metrics(self, eval_preds):
        """
        Compute evaluation metrics for token classification tasks.

        This function calculates precision, recall, F1 score, and accuracy
        for the predictions made by a token classification model.

        Args:
            eval_preds (tuple): A tuple containing model predictions and true labels.
                The first element is the logits output by the model,
                and the second element is the true labels.

        Returns:
            dict: A dictionary containing the computed metrics:
                - 'precision': The overall precision score.
                - 'recall': The overall recall score.
                - 'f1': The overall F1 score.
                - 'accuracy': The overall accuracy score.
        """
        logits, labels = eval_preds

        predictions = np.argmax(logits, axis=-1)
        true_labels = [[self.label_names[l] for l in label if l !=-100] for label in labels]

        true_predictions = [[self.label_names[p] for p, l in zip(prediction, label) if l !=-100] for prediction, label in zip(predictions, labels)]

        all_metrics = self.metric.compute(predictions= true_predictions, references = true_labels)

        return {
            'precision':all_metrics['overall_precision'],
            'recall':all_metrics['overall_recall'],
            'f1':all_metrics['overall_f1'],
            'accuracy':all_metrics['overall_accuracy']
        }
