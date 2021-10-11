import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from .labeling import remove_iob_preffix

def compute_metrics(predictions, id2tag):
    pred, labels = predictions
    pred = np.argmax(pred, axis=2)

    # WARNING! These contain non-valid true labels for ignored sentencepiece tags (-100 tags),
    # which should NOT be used when calculating metrics!
    trues_preds_flattened = list(zip(
        predictions.label_ids.flatten(),
        np.argmax(predictions.predictions, axis=2).flatten()
    ))
    # Metrics should only be calculated on true labels != -100
    # (huggingface ignores these for the loss, since theyre extra sentencepiece toks of a word)
    # 
    # Quote:
    #      https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities
    # 
    #          One way to handle this is to only train on the tag labels for the first subtoken of
    #          a split token. We can do this in 🤗 Transformers by setting the labels we wish to ignore to -100.
    #          In the example above, if the label for @HuggingFace is 3 (indexing B-corporation), we would set
    #          the labels of ['@', 'hugging', '##face'] to [3, -100, -100].
    #
    trues_preds_flattened_valid = [(true, pred) for true, pred in trues_preds_flattened if true != -100]

    true_valid, pred_valid = list(zip(*trues_preds_flattened_valid))
    
    true_valid_tags_no_iob = [remove_iob_preffix(id2tag[tag]) for tag in true_valid]
    pred_valid_tags_no_iob = [remove_iob_preffix(id2tag[tag]) for tag in pred_valid]
    
    
    accuracy = accuracy_score(y_true=true_valid, y_pred=pred_valid)
    recall = recall_score(y_true=true_valid, y_pred=pred_valid, average='macro')
    precision = precision_score(y_true=true_valid, y_pred=pred_valid, average='macro')
    f1_macro = f1_score(y_true=true_valid, y_pred=pred_valid, average='macro')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro
    } 

