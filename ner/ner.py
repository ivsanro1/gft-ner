import numpy as np
import torch

from .labeling import remove_iob_preffix_list
from .tokenize import tags_reduce

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # XXX: Should be the same. Check
        if self.labels is not None:
            return len(self.labels)
        else:
            k = list(self.encodings.keys())[0]
            return len(self.encodings[k])



def encode_tags(tags, encodings, tag2id):
    '''
    Utility function to encode the `tags` given a sentencepiece encoding `encodings`.
    `encodings` will contain non-root sentencepiece tokens (e.g. #isima), and these will
    be mapped to the tag -100 so it is ignored by the loss function of huggingface transformers
    '''
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        try: 
            # set labels whose first offset position is 0 and the second is not 0
            # print(f'arr_offset.shape: {arr_offset.shape}')
            # print(f'len(doc_labels): {len(doc_labels)}')
            # print(f'len(doc_enc_labels): {len(doc_enc_labels)}')
            # print(np.unique((arr_offset[:,0] == 0) & (arr_offset[:,1] != 0), return_counts=True))
            # print()
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            # assert False
        except ValueError as e: 
            print(e)
            print('this doc_enc_labels will be -100 filled')
            # Ran into the following bug: 
            # "ValueError: NumPy boolean array indexing assignment cannot 
            # assign 411 input values to the 387 output values 
            # where the mask is true"
            # skipping sample to avoid further time sink
            # returning the doc_enc_labels filled with -100 values
            pass 
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels
    
    
    
def add_sentencepiece_data_to_df(df, encodings, offset_mapping, labels, tokenizer=None):
    '''
    Adds all preprocessed data related to sentencepiece tokenization and labels to the samples' dataframe

    `df`: ner pd.DataFrame
    `encodings`: obtained from Huggingface tokenizer. Same length as `df`
    `labels`: List of sequences of sentencepiecelabels for each sentence (row in the df). Same length as `df`
    '''

    df['sp_input_ids'] = encodings['input_ids']
    df['sp_token_type_ids'] = encodings['token_type_ids']
    df['sp_attention_mask'] = encodings['attention_mask']
    df['sp_offset_mapping'] = offset_mapping
    if labels is not None:
        df['sp_labels'] = labels
    
    if tokenizer != None:
        df['sp_input_tok'] = [[tokenizer._convert_id_to_token(tok_id) for tok_id in sent] for sent in df['sp_input_ids']]

    return df



SPECIAL_SP_TOKS = '[SEP]', '[CLS]', '[PAD]'
def get_token_predictions_from_sentencepiece_token_predictions(toks, sp_toks, sp_pred, offset_mapping):
    '''
    Using offset_mappings, maps the preds made to the sentencepiece tokens original non-sentencepiece tokens
    
    toks:    df_ner['ner_list_toks']
             ['de', 'Bilbao', 'procedieron', ...]
    
    sp_toks: df_ner['sp_input_tok']
             ['[CLS]', 'de', 'Bilbao', 'procedi', '##eron', ...]
    
    sp_pred: df_ner['ner_predict_y_hat_label']
             ['[CLS]', 'O', 'LOC', 'O', 'O', ...]
             
    offset_mapping: df_ner['sp_offset_mapping']
             ['O', 'LOC', 'O', 'O', ...]
             
             
             
    
    example input and output:
            sp toks:      '[CLS]', 'de', 'Bilbao', 'procedi', '##eron'
            sp pred:      'O'    , 'O' , 'LOC'   , 'O'      , 'O' 
        
            not sp toks:  'de', 'Bilbao', 'procedieron'
            not sp pred:  'O' , 'LOC'   , 'O'          
    '''
    list_list_toks_preds = [] # List of all sentencepiece predictions of the non sentencepiece token
    current_token_preds = []
    current_token_confidences = [] # TODO...
    for i in range(len(offset_mapping)):
        if offset_mapping[i][0] == 0: # Word starter
            # End current accumulation and add (if there's any)
            if current_token_preds != []:
                list_list_toks_preds.append(current_token_preds)
                current_token_preds = []
            if sp_toks[i] not in SPECIAL_SP_TOKS: # Not special token, start acum
                current_token_preds.append(sp_pred[i])
            else: # Special token, skip start
                pass
        else: # Not a word starter, keep acum
            current_token_preds.append(sp_pred[i])
    
    # Print warning instead of assert
    # Happens when sentence is too long
    # ipdb> len(toks)
    # 495
    # ipdb> len(list_list_toks_preds)
    # 430
    # assert len(toks) == len(list_list_toks_preds)
    if len(toks) != len(list_list_toks_preds):
        print(f'''WARNING: original sequence length (len(toks)={len(toks)}) and length of backward-obtained labels '''
              f'''from sentencepiece-level predictions (len(list_list_toks_preds)={len(list_list_toks_preds)}) mismatch.\n'''
              f'''This is probably due to the sequence being too long (near 512 tokens) and sentencepieced tokenization exceeded 512.\n'''
              f'''Extending predictions with 'O' to match length...\n''')
        len_diff = len(toks) - len(list_list_toks_preds)
        list_list_toks_preds = list_list_toks_preds + ['O'] * len_diff
        assert len(list_list_toks_preds) == len(toks)

    return list_list_toks_preds





def get_tag2id_id2tag(df):
    '''
    Gets mappings of tag2id and id2tag given the df of ner
    `df`: df_al (df_ner)
    '''
    unique_tags = set(tag for doc in df['ner_list_tags'] for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    return {
        'tag2id': tag2id,
        'id2tag': id2tag
    }




def reduce_sptok_preds_to_tok_preds(df):
    '''
    For every non-sentencepiece token, creates its list of correspondent sentencepiece predictions
    and reduces it following the logic of `tags_reduce`
    '''
    # Sets each non-sentencepiece token prediction as the list
    # of predictions of the sentencepiece tokens it's formed of.
    # It is intended that you perform a reduction operation afterwards (`tags_reduce`)
    df['list_sp_preds_per_tok'] = df.apply(
        lambda row: get_token_predictions_from_sentencepiece_token_predictions(
            toks=row['ner_list_toks'],
            sp_toks=row['sp_input_tok'],
            sp_pred=row['ner_predict_y_hat_label'],
            offset_mapping=row['sp_offset_mapping']
    ), axis=1)

    # `lspppt` stands for 'List of SentencePiece Predictions Per Token'
    # It is the list of sentencepiece predictions to be token-wise reduced, for each non-sentencepiece token
    df['list_preds_tok'] = df['list_sp_preds_per_tok'].progress_apply(lambda lspppt: [tags_reduce(lp) for lp in lspppt])
    return df






def get_char_labels_for_pylighter_from_prediction(txt, list_spans, list_preds_tok):
    '''
    Auto-annotation function that sets the char-level tags needed for PyLighter using
    some non-sentencepiece tokens
    
    It creates char-level annotations given the token-level predictions of NER.
    Then, it chunks char-level annotations together if they're separated by certain characters (like spaces)
        This is done because spaces cannot be predicted by NER, since NER is done on the tokens, but we are interested in
        potentially joining the annotations
    Then, it adds the I- or B- prefix to the tags, depending on a simple heuristic
        B- if it starts
        I- if it continues
    '''
    assert len(list_spans) == len(list_preds_tok)
    
    # Initialize list
    labels_char = ['O' for i in range(len(txt))]
    
    # Remove IOB
    list_preds_tok_no_iob = remove_iob_preffix_list(list_preds_tok)
    assert len(list_preds_tok_no_iob) == len(list_preds_tok)
    
    for i in range(len(list_preds_tok_no_iob)):
        span = list_spans[i]
        label = list_preds_tok_no_iob[i]
        for j in range(*span):
            labels_char[j] = label
            
            
    # Chunk (join) if separated only by space and if it's the same label
    acum = []
    last_label = None
    for i in range(len(txt)):
        c = txt[i]
        l = labels_char[i]
        if l == 'O' and c == ' ':
            if last_label is not None:
                acum.append(i)
        if l == 'O' and c != ' ':
            acum = []
        if l != 'O':
            if acum != []:
                for idx in acum:
                    labels_char[idx] = last_label
            last_label = l
            
            
    # Make them IOB now
    start_entity = True
    for i in range(len(txt)):
        l = labels_char[i]
        if l == 'O':
            start_entity = True
            continue
        elif l != 'O' and start_entity: # We start: Append B-
            start_entity = False
            labels_char[i] = 'B-' + labels_char[i]
        elif l != 'O' and not start_entity: # We continue: Append I-
            start_entity = False
            labels_char[i] = 'I-' + labels_char[i]
        else:
            assert False


    assert len(txt) == len(labels_char)
            
    return labels_char


def set_df_ner_char_labels_pylighter_from_prediction(df):
    '''
    `list_preds_tok` is obtained by the function `reduce_sptok_preds_to_tok_preds`
    '''
    df['ner_char_labels_pylighter'] = df.apply(
        lambda row: get_char_labels_for_pylighter_from_prediction(
            row['txt'],
            row['d_ret_tokenize_w_labels']['list_spans'],
            row['list_preds_tok']
        ),
        axis=1)
    return df