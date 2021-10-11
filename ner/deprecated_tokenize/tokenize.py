# DEPRECATED
'''
import nltk

from .abbv import set_base_nltk_abbv_spanish, d_spanish_abbv_case_insensitive

extra_abbreviations = set([k.strip('.') for k in d_spanish_abbv_case_insensitive.keys()])

# Since we already load the Spanish tokenizer as base, we do not need to specify language when we call .tokenize(text)
sentence_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
sentence_tokenizer._params.abbrev_types = set.union(
    set_base_nltk_abbv_spanish,
    extra_abbreviations
)


'''