import spacy

nlp = spacy.load("es_core_news_sm")

def spacy_sent_tokenize(txt, spans=True):
    doc = nlp(txt)
    list_sents = [sent for sent in doc.sents]
    if spans:
        list_spans = []
        for sent in list_sents:
            start_tok = sent[0]
            start_idx = start_tok.idx
            end_tok = sent[-1]
            end_idx = end_tok.idx + len(end_tok)
            list_spans.append((start_idx, end_idx))
        return list_spans
    else:
        return [sent.text for sent in list_sents]


from .abbv import d_spanish_abbv_lower
def fix_sent_tokenization(txt, spans, ending_abbreviations, verbose=False):
    '''
    Joins the sentences of `txt` delimited by `spans`.
    It joins the span `i` with the span `i+1` if span `i` ends with any of the
    provided `ending_abbreviations`

    One example of `ending_abbreviations` can be `d_spanish_abbv_lower.keys()`
    '''
    for i in reversed(range(len(spans)-1)): # Traverse the list of spans (but last span) backwards
        span = spans[i] # start,end tuple
        
        span_txt = txt[span[0]:span[1]].lower()
        if verbose:
            print(f'span_txt: {span_txt}')
            for k in d_spanish_abbv_lower.keys():
                print(f'k: {k}')
                print(f'''span_txt.endswith(' ' + k): {span_txt.endswith(' ' + k)}''')
                print(f'''span_txt.endswith('.' + k): {span_txt.endswith('.' + k)}''')
            print()
        if any([span_txt.endswith(' ' + k) or span_txt.endswith('.' + k) for k in d_spanish_abbv_lower.keys()]):
            if verbose: print(f'Appended {i} with {i+1}')
            span_next = spans.pop(i+1)
            spans[i] = (span[0], span_next[1])

    return spans


from nltk.tokenize import word_tokenize
def join_short_sentences(txt, spans, tokenizer, thr_n_sptoks=500, verbose=False):
    '''
    RECOMMENDATION: txt should NOT be all uppercase or tokenizer will split it into a bunch of sentnecepiece tokens    
    '''
    offset = 0
    for i in range(len(spans)-1): # Traverse the list of spans (but last span) backwards
        i = i - offset
        span = spans[i] # start,end tuple
        span_txt = txt[span[0]:span[1]]
        n_sptoks = len(tokenizer(span_txt)['input_ids'])

        span_next = spans[i+1]
        span_next_txt = txt[span_next[0]:span_next[1]]
        n_sptoks_next = len(tokenizer(span_next_txt)['input_ids'])
        
        if verbose:
            print(f'i: {i}')
            print(f'n_sptoks: {n_sptoks}')
            print(f'n_sptoks_next: {n_sptoks_next}')
        if n_sptoks + n_sptoks_next < thr_n_sptoks:
            if verbose:
                print(f'n_sptoks + n_sptoks_next (={n_sptoks + n_sptoks_next}) < thr_n_sptoks ({thr_n_sptoks})')
                print(f'Appended {i} with {i+1}')
            span_next = spans.pop(i+1)
            spans[i] = (span[0], span_next[1])
            offset += 1
        else:
            if verbose:
                print(f'n_sptoks + n_sptoks_next (={n_sptoks + n_sptoks_next}) NOT < thr_n_sptoks ({thr_n_sptoks})')
                print('Limit reached, not accumulating')

    return spans


def spacy_word_tokenize(txt, spans=True):
    doc = nlp(txt)
    
    if spans:
        list_idx = [tok.idx for tok in doc]
        list_lens = [len(tok.text) for tok in doc]
        list_spans = [(list_idx[i], list_idx[i] + list_lens[i]) for i in range(len(list_idx))]
        return list_spans
    else:
        return [tok.text for tok in doc]
    

def tags_reduce(list_tags):
    set_tags = set(list_tags)
    assert len(set_tags) > 0
    b_tags = [el for el in set_tags if el.startswith('B-')]
    i_tags = [el for el in set_tags if el.startswith('I-')]

    if len(b_tags) > 0: # B tags first
        return b_tags[0] # 1st found
    elif len(i_tags) > 0: # I tags afterwards
        return i_tags[0] # 1st found
    else: # nothing found → 'O' tag
        return list(set_tags)[0]

    
def tokenize_with_ner_labels_from_pylighter(txt, labels, fn_word_span_tokenize, reduce_tags=True, training=True):
    '''
    txt
        plain text that was annotated
    labels
        char-wise labels of annotated text. Obtained with PyLighter
    fn_word_span_tokenize
        function that tokenizes a text, returning the span tuples in a list e.g. [(0,5), (6,12), ...]
        e.g.: `partial(spacy_word_tokenize, spans=True)`
    reduce_tags
        whether to apply a reduce fn to the list of each token's list of char tags
    '''
    if training:
        assert len(txt) == len(labels) # one label per char (PyLighter output)
    
    if training and labels is None:
        raise ValueError('if training == True, you are expected to pass labels != None')
    if not training and labels is not None:
        print('WARN: [tokenize_with_ner_labels_from_pylighter] training == False, but you are passing labels != None. Do you want to train?')

    token_spans = fn_word_span_tokenize(txt)
    
    list_toks = []
    list_tags = []
    
    for span in token_spans:
        start, end = span
        span_text = txt[start:end] # MAYBE APPLY INVALID TOKEN LOGIC HERE?
        if training:
            span_tags = labels[start:end]
        
        list_toks.append(span_text)
        
        if training:
            if reduce_tags:
                list_tags.append(tags_reduce(span_tags))
            else:
                list_tags.append(span_tags)
    
    assert len(list_toks) == len(token_spans)
    if training:
        assert len(list_toks) == len(list_tags)
        
    d_ret = {
        'list_toks': list_toks,
        'list_spans': token_spans
    }
    if training:
        d_ret['list_tags'] = list_tags

    return d_ret


def clear_invalid_tokens_kept_by_spacy(d_ret_tokenize_w_labels):
    '''
    Spacy will keep some tokens that are not interesting.
    
    These tokens are troublesome later when sentencepiece-tokenizing,
    because they won't be kept, and the labels of them are kept in a parallel list,
    causing a length mismatch.
    
    These tokens are '\n', ' ', etc.
    This functions removes them.
    
    `d_ret_tokenize_w_labels` is obtained with the function `tokenize_with_ner_labels_from_pylighter`
    '''
    # Create empty list that will be filled if tokens are valid
    d_ret = {k:[] for k in d_ret_tokenize_w_labels.keys()}
    
    training = 'list_tags' in d_ret_tokenize_w_labels
    
    list_toks = d_ret_tokenize_w_labels['list_toks']
    list_spans = d_ret_tokenize_w_labels['list_spans']
    if training:
        list_tags = d_ret_tokenize_w_labels['list_tags']
    
    for i in range(len(list_toks)):
        tok = list_toks[i]
        span = list_spans[i]
        if training:
            tag = list_tags[i]
        
        if tok.strip() == '': # invalid            
            pass
        else: # valid
            d_ret['list_toks'].append(tok)
            d_ret['list_spans'].append(span)
            if training:
                d_ret['list_tags'].append(tag)
            
    assert len(d_ret['list_toks']) == len(d_ret['list_spans'])
    if training:
        assert len(d_ret['list_tags']) == len(d_ret['list_toks'])

    return d_ret