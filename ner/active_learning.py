import numpy as np
from .ner import SPECIAL_SP_TOKS

def mnlp(ner_row):
    '''
    Params:
        `ner_row` = df_ner_ev.iloc[X]
    
    
    Explanation:
        Calculates the Maximum Normalized Log-Probability (MNLP) of a sample with the predictions.
        i.e. the average log-proba output of the model.
        Intuitively, a log-proba can be seen as the confidence of the model
        in its prediction, with the interesting aspect that it is always negative, or 0.

        When the model is 100% sure of its prediction, the log-proba is 0
        As the confidence of the model in its prediction goes down, the
        log-proba falls too, going negative.
        
        A log-proba of 0.0 is -inf, but it is impossible that it happens in
        the practice since we're playing with softmaxed values (distros that
        HAVE TO sum to 1).
        
        With respect to Active Learning sample selection, we are interested
        in getting samples from the pool that minimize this value
    
    Reference:
        Eq. (3) @ DEEP ACTIVE LEARNING FOR NAMED ENTITY RECOGNITION
        https://arxiv.org/pdf/1707.05928.pdf
    
    Examples (Note: `IN` is not this function's inputs):
        IN: [1.0, 1.0, 1.0]
        OUT: 0.0

        IN: [0.9, 0.9, 0.9]
        OUT: -0.10536051565782628

        IN: [0.5, 0.5, 0.9]
        OUT: -0.49721829225923897
    '''
    if len(ner_row['ner_list_toks']) == 0:  # Empty input sequence
        return 0

    # Num sp tokens and num of output probas is the same
    assert len(ner_row['ner_predict_softmax']) == len(ner_row['sp_input_tok'])
    
    n_sp_toks = len(ner_row['ner_predict_softmax'])
    
    # Keep y_hat_probas of valid sentencepiece tokens. Non-valid are [CLS], [SEP], and [PAD]
    l_softmaxes = np.array([
        ner_row['ner_predict_softmax'][i] for i in range(n_sp_toks) if ner_row['sp_input_tok'][i] not in SPECIAL_SP_TOKS
    ])
    
    l_max_softmaxes = np.max(l_softmaxes, axis=1)
    
    mnlp = sum(np.log(l_max_softmaxes)) / len(l_max_softmaxes)
    return mnlp