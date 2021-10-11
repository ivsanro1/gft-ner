import numpy as np
import pickle
from itertools import chain
from functools import partial

from tqdm.notebook import tqdm
from sklearn.metrics import classification_report

from .tokenize import (
    spacy_word_tokenize,
    tokenize_with_ner_labels_from_pylighter,
    clear_invalid_tokens_kept_by_spacy
)

from .ner import (
    NERDataset,
    encode_tags,
    add_sentencepiece_data_to_df,
    get_token_predictions_from_sentencepiece_token_predictions,
    get_tag2id_id2tag
)

from .labeling import remove_iob_preffix





def full_ner_preproc(df_sents, tokenizer, training, dir_ner, d_tag2id_id2tag=None):
    '''
    df_sentences: df with 'txt' col with the sents
    tokenizer: huggingface trasnformers' tokenizer. Used for text tokenization and encoding
    training: Bool.
        True: If it's a preproc for training (with labels) or
        False: If it's for inference (without labels)
    '''
    
    if not training and d_tag2id_id2tag is None:
        raise ValueError('If training == False, you have to specify a d_tag2id_id2tag')

    fn_word_span_tokenize = partial(spacy_word_tokenize, spans=True)


    df_sents['d_ret_tokenize_w_labels'] = df_sents.progress_apply(
        lambda row: clear_invalid_tokens_kept_by_spacy(
            tokenize_with_ner_labels_from_pylighter(
                txt=row['txt'],
                labels=row['ner_char_labels_pylighter'] if training else None,
                fn_word_span_tokenize=fn_word_span_tokenize,
                training=training
            )
        ), axis=1)
    
    df_sents['ner_list_toks'] = df_sents['d_ret_tokenize_w_labels'].apply(lambda d: d['list_toks'])
    
    if training:
        df_sents['ner_list_tags'] = df_sents['d_ret_tokenize_w_labels'].apply(lambda d: d['list_tags'])
        
    # Create if it was not passed
    if d_tag2id_id2tag is None:
        d_tag2id_id2tag = get_tag2id_id2tag(df_sents)
        
        # Serialize tag2id and id2tag
        path_d_tag2id_id2tag = dir_ner / 'd_tag2id_id2tag.pkl'
        with open(path_d_tag2id_id2tag, 'wb') as f:
            pickle.dump(d_tag2id_id2tag, f)
    
    
    tag2id = d_tag2id_id2tag['tag2id']
    id2tag = d_tag2id_id2tag['id2tag']
        
    # ENCODE 
    encodings = tokenizer(
        df_sents['ner_list_toks'].tolist(),
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    
    
    print(f"Max len tok seq is {max([len(ltoks) for ltoks in df_sents['ner_list_toks']])} toks long")
    print(f"Sents are being padded to {len(encodings[0])} sentencepiece toks long")

    if training:
        # GET TOKEN TAGS
        word_tags = df_sents['ner_list_tags'].tolist()
        
        # TO SENTENCEPIECE
        sp_tags = encode_tags(word_tags, encodings, tag2id)

    # KEEP SENTENCEPIECE OFFSET MAPPINGS
    offset_mapping = encodings.pop("offset_mapping") # we don't want to pass this to the model
    
    if training:
        ner_dataset_obj = NERDataset(encodings, sp_tags)
    else:
        ner_dataset_obj = NERDataset(encodings, labels=None)

    df_sents = add_sentencepiece_data_to_df(
        df=df_sents,
        encodings=encodings,
        offset_mapping=offset_mapping,
        labels=sp_tags if training else None,
        tokenizer=tokenizer
    )
    
    d_ret = {
        'df_sents': df_sents,
        'dataset_obj': ner_dataset_obj,
    }

    if training:
        d_ret['d_tag2id_id2tag'] = d_tag2id_id2tag
        d_ret['tag2id'] = tag2id # replicated from d_tag2id_id2tag just for convenience
        d_ret['id2tag'] = id2tag # replicated from d_tag2id_id2tag just for convenience
        d_ret['unique_tags'] = list(tag2id.keys())

    return d_ret


from scipy.special import softmax
def predict_single_sample(d_sample, model_ner, id2tag):
    '''
    Makes a full prediction of a single sample.
    
    `d_sample`: dict of the sample to be predicted. This is an item from NERDataset.
                This dict has to have 'input_ids' and 'attention_mask' attributes,
                which are obtained with a Huggingface tokenizer.
                
    `model_ner`: Trained NER model from Huggingface trasnformers (XForTokenClassification)
    '''
    ner_output = model_ner.forward(
        input_ids=d_sample['input_ids'].unsqueeze(0),
        attention_mask=d_sample['attention_mask'].unsqueeze(0),
    )
    d_ret = {}
    d_ret['logits'] = ner_output['logits'][0].detach().numpy()
    d_ret['softmax'] = softmax(d_ret['logits'], axis=1)
    d_ret['y_hat_id'] = [np.argmax(word_softmax) for word_softmax in d_ret['softmax']]
    d_ret['y_hat_label'] = [id2tag[id_argmax] for id_argmax in d_ret['y_hat_id']]
    return d_ret       


def predict_entire_dataset(d_ner_preproc, model_ner, id2tag):
    '''
    Performs full prediction of a `d_ner_preproc` object, which contains the `df_ner` and the `dataset_obj` (`NERDataset`)
    The `d_ner_preproc` is obtained 
    '''
    df = d_ner_preproc['df_sents']
    
    df['d_ner_predict'] = [
        predict_single_sample(
            sample,
            model_ner=model_ner,
            id2tag=id2tag
        ) for sample in tqdm(d_ner_preproc['dataset_obj'])
    ]
    
    df = post_process_predictions(df)
    return df


def post_process_predictions(df):
    df['ner_predict_logits'] = df['d_ner_predict'].apply(lambda d: d['logits'])
    df['ner_predict_softmax'] = df['d_ner_predict'].apply(lambda d: d['softmax'])
    df['ner_predict_y_hat_id'] = df['d_ner_predict'].apply(lambda d: d['y_hat_id'])
    df['ner_predict_y_hat_label'] = df['d_ner_predict'].apply(lambda d: d['y_hat_label'])
    return df



def ner_evaluation_report(df, at_sp_level=False, id2tag=None):
    '''
    You must have done `reduce_sptok_preds_to_tok_preds(df)` before.
    Otherwise, the assert won't pass because sentencepiece seq len != non-sentencepiece seq len
    
    # Will fail
    >>> assert len(flattened_labels_pred) == len(flattened_labels_true)
    '''
    if not at_sp_level:
        flattened_labels_pred = list(chain.from_iterable(df['list_preds_tok']))
        flattened_labels_true = list(chain.from_iterable(df['ner_list_tags']))
    else:
        if id2tag is None:
            raise ValueError('if `ner_evaluation_report` is done at sp level (at_sp_level=True), you must pass an id2tag.')
        flattened_labels_pred = list(chain.from_iterable(df['ner_predict_y_hat_label']))
        flattened_labels_true = list(chain.from_iterable(df['sp_labels']))
        flattened_labels_true = [id2tag.get(t) for t in flattened_labels_true]

        flattened_labels_pred = np.array(flattened_labels_pred)
        flattened_labels_true = np.array(flattened_labels_true)

        # Assert before non-root sp removal (#sptok)
        assert len(flattened_labels_pred) == len(flattened_labels_true)
        idx_valid = np.where(flattened_labels_true != None)
        flattened_labels_pred = flattened_labels_pred[idx_valid]
        flattened_labels_true = flattened_labels_true[idx_valid]


    # ASSERT SHOULD PASS NOW THAT WE TRANSFORMED SENTENCEPIECE PREDS TO NON-SENTENECEPIECE PREDS
    assert len(flattened_labels_pred) == len(flattened_labels_true)

    flattened_labels_pred_no_preffix = [remove_iob_preffix(tag) for tag in flattened_labels_pred]
    flattened_labels_true_no_preffix = [remove_iob_preffix(tag) for tag in flattened_labels_true]

    print(classification_report(flattened_labels_pred_no_preffix, flattened_labels_true_no_preffix))





def _get_spans_ner_positives(list_tok_spans, list_tok_preds):
    '''
    list_tok_spans: d_ret_tokenize_w_labels['list_spans']
    list_tok_preds: list_preds_tok
    
    TODO: For now, I/B prefixes are being ignored. Maybe we should not? Would it make any sense?
    '''
    assert len(list_tok_spans) == len(list_tok_preds)
    
    list_tok_preds = [remove_iob_preffix(pred) for pred in list_tok_preds]
    
    list_d_ner_predictions = []
    
    start = None
    end = None
    pred_to_add = None
            
    for i in range(len(list_tok_spans)):
        pred = list_tok_preds[i]
        span = list_tok_spans[i]
        if start is None and pred != 'O': # We have not started an acum. Start acum since pred != 'O'
            pred_to_add = pred
            start = span[0]
        elif start is not None and pred != 'O': # Already started an acum. Continue acum since pred != 'O'
            # Maybe is a different tag? (for the future)
            if pred_to_add != pred: # Different tag encountered. Finish last acum but start acumulating again
                # Lookbehind addition, because this span is from new, different class. We must use last span
                last_span = list_tok_spans[i-1]
                end = last_span[1] 
                d_to_add = {
                    'entity': pred_to_add,
                    'start': start,
                    'end': end
                }
                list_d_ner_predictions.append(d_to_add)
                
                # Reinitialize in this very iteration, because we must start acumulating already!
                start = span[0]
                end = None
                pred_to_add = pred # Remember to set new pred
            else: # Same tag encountered. Keep acumulating (pass)
                pass
        elif start is not None and pred == 'O': # We have started an acum. We end acum since pred == 'O'. Add end and set start=None to indicate we are not acumulating
            # Lookbehind addition, because this span is from 'O'. We must use last span
            last_span = list_tok_spans[i-1]
            end = last_span[1] 
            d_to_add = {
                'entity': pred_to_add,
                'start': start,
                'end': end
            }
            list_d_ner_predictions.append(d_to_add)
            start = None
            end = None
            pred_to_add = None
        elif start is None and pred == 'O': # We have not started an acum. We do not start acum since pred == 'O'. Simply pass
            pass
        
    # Remember to add last if we get out of the loop and we are still acumulating!
    if start is not None:
#         print('Adding final')
        end = list_tok_spans[-1][1] # last span. End of sequence
        d_to_add = {
            'entity': pred_to_add,
            'start': start,
            'end': end
        }
        list_d_ner_predictions.append(d_to_add)
    return list_d_ner_predictions


def _test_get_spans_ner_positives():
    list_spans = [(0, 5), (6, 10), (11, 17), (18, 21), (22, 24), (25, 31), (32, 34), (35, 43), (44, 46), (47, 58), (59, 64), (65, 67), (68, 75), (76, 80), (80, 81), (82, 90), (91, 93), (94, 102), (103, 110), (111, 113), (114, 116), (117, 119), (120, 127), (128, 133), (134, 136), (137, 139), (140, 142), (143, 151), (152, 153), (154, 156), (157, 167), (168, 170), (171, 173), (174, 182), (183, 186), (187, 190), (191, 198), (198, 199), (200, 202), (203, 209), (210, 212), (213, 217), (218, 219), (219, 231), (232, 235), (236, 245), (245, 246), (246, 247)]
    list_preds_tok = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC_OFICINA', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC_OFICINA', 'I-LOC_OFICINA', 'I-LOC_OFICINA', 'I-LOC_OFICINA', 'I-LOC_OFICINA', 'I-LOC_OFICINA', 'I-LOC_OFICINA', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    
    expected_ret = [
        {'entity': 'LOC_OFICINA', 'start': 76, 'end': 80},
        {'entity': 'LOC_OFICINA', 'start': 117, 'end': 151}
    ]
          
    assert _get_spans_ner_positives(list_spans, list_preds_tok) == expected_ret
    
_test_get_spans_ner_positives()


def set_spans_ner_positives_all_df_infer(df):
    list_ner_final_predictions = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        ner_final_predictions = _get_spans_ner_positives(
            row['d_ret_tokenize_w_labels']['list_spans'],
            row['list_preds_tok']
        )
        
        # Add extra interesting data to final predictions
        for d_pred in ner_final_predictions:
            start = d_pred['start']
            end = d_pred['end']
            d_pred['text'] = row['txt'][start:end]
        
        list_ner_final_predictions.append(ner_final_predictions)
    df['ner_final_predictions'] = list_ner_final_predictions

    return df




