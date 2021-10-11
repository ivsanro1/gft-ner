# DEPRECATED?
'''
from .tokenize import sentence_tokenizer
from .abbv import d_spanish_abbv_case_insensitive

def get_last_word(sent):
    return sent.split(' ')[-1]

def _proc_ending(w):
    return w.lower().strip('.()/[]')


# This list will contain the ending words that are not valid, so the two split sentences will become joined again
BAD_SENT_ENDINGS = set([_proc_ending(k) for k in d_spanish_abbv_case_insensitive.keys()])



def sentencify(txt):
    return fix_sentence_tokenization(sentence_tokenizer.tokenize(txt))


def fix_sentence_tokenization(list_sents, verbose=False):
    ''''''
    Since adding abbreviations (.tokenize.py) does not seem to work when after the
    abbreviation there is a capitalized word. E.g.:

    >>> sentence_tokenizer.tokenize('... un importe en la oficina con direccion Avda. Juan Carlos I, 82 300007 Murcia, de ...')

    >>> Out: ['... un importe en la oficina con direccion Avda.',
    >>>       'Juan Carlos I, 82 300007 Murcia, de ...']


    It will be much easier and less of a hassle to just fix the wrongly split sentences given a basic heuristic
    ''''''

    fixed_list_sents = []
    sent_acum = ''
    
    def _acum_sent_w_space_if_join(sent_acum, sent_to_add):
        if sent_acum == '':
            if verbose: print('\tstart acum')
            sent_acum = sent_to_add
        else:
            if verbose: print('\tadding')
            sent_acum += ' ' + sent_to_add # Add space between rejoined sentences
        return sent_acum

    for i, sent in enumerate(list_sents):
        if _proc_ending(get_last_word(sent)) in BAD_SENT_ENDINGS and i+1 < len(list_sents): # Keep acumulating the sentence because we found a bad ending
            if verbose: print(f'bad ending: {_proc_ending(get_last_word(sent))}')
            sent_acum = _acum_sent_w_space_if_join(sent_acum, sent)
        else: # We finished acumulating. Add current and append
            if not i+1 < len(list_sents):
                if verbose: print('Finished')
            else:
                if verbose: print(f'Not finished but good ending: {_proc_ending(get_last_word(sent))}')
            if verbose: print('Finishing acum and appending...')
            
            sent_acum = _acum_sent_w_space_if_join(sent_acum, sent)

            fixed_list_sents.append(sent_acum)
            sent_acum = ''

    return fixed_list_sents

'''