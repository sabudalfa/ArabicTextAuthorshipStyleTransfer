import evaluate
from simpleio import *
import arabic_reshaper
from matplotlib import font_manager
from bidi import algorithm as bidialg

EXPERIMENT_PATH = '../output/experiments/30 - author-classification/output/Meta-Llama-3-8B/'
model_name = EXPERIMENT_PATH.split('/')[-2]
# predictions = read_from_file('../output/experiments/31 - author-classification with roberta/predicted_authors.json')
# references = read_from_file('../output/experiments/31 - author-classification with roberta/ground_truth_authors.json')

predictions = read_from_json_file(f'{EXPERIMENT_PATH}/predicted_authors.json')
references = read_from_json_file(f'{EXPERIMENT_PATH}/ground_truth_authors.json')

legal_authors = set(references)
samples = [
    (prediction, reference)
    for prediction, reference in zip(predictions, references)
    if prediction in legal_authors
]
predictions = [prediction for prediction, _ in samples]
references = [reference for _, reference in samples]

authors = list(set(predictions + references))
print(authors)
print(list(sorted(set(predictions))))
print(list(sorted(set(references))))
print(len(authors))

int_predictions = [authors.index(p) for p in predictions]
int_references = [authors.index(r) for r in references]

accuracy_metric = evaluate.load('accuracy')
recall_metric = evaluate.load('recall')
precision_metric = evaluate.load("precision")
f1_metric = evaluate.load('f1')

evaluation_dict = {
    'accuracy': accuracy_metric.compute(references=int_references, predictions=int_predictions)['accuracy'],
    'recall-micro': recall_metric.compute(references=int_references, predictions=int_predictions, average='micro')['recall'],
    'recall-macro': recall_metric.compute(references=int_references, predictions=int_predictions, average='macro')['recall'],
    'precision-micro': precision_metric.compute(references=int_references, predictions=int_predictions, average='micro')['precision'],
    'precision-macro': precision_metric.compute(references=int_references, predictions=int_predictions, average='macro')['precision'],
    'f1-micro': f1_metric.compute(references=int_references, predictions=int_predictions, average='micro')['f1'],
    'f1-macro': f1_metric.compute(references=int_references, predictions=int_predictions, average='macro')['f1'],
}

write_to_json_file(
    f'{EXPERIMENT_PATH}/evaluation.json',
    content=evaluation_dict,
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(y_true, y_pred, class_names=None, font='DejaVu Sans'):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the plot
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                     xticklabels=[
                         bidialg.get_display(arabic_reshaper.reshape(class_name))
                         for class_name in class_names
                     ],
                     yticklabels=[
                         bidialg.get_display(arabic_reshaper.reshape(class_name))
                         for class_name in class_names
                     ],
                     linewidths=1,
                     linecolor='white',
                     )

    # Add labels and title
    ax.set_xlabel("Predicted Author", fontsize=14, labelpad=10)
    ax.set_ylabel("Actual Author", fontsize=14, labelpad=10)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=16, pad=15)

    # Rotate tick labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


print(evaluation_dict)

plot_confusion_matrix(y_true=references, y_pred=predictions, class_names=list(set(references)))
