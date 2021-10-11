import re
from pylighter import Annotation

def dump_annot(annotation, df):
    '''
    annotation.corpus will have the texts
    annotation.labels will habe the labels
    '''
    assert len(annotation.corpus == len(df))
    assert all(annotation.corpus == df['txt']) # All texts are the same?
    df['ner_char_labels_pylighter'] = annotation.labels
    return df


def load_annot(df, additional_label_names=None):
    assert 'ner_char_labels_pylighter' in df

    if additional_label_names is None: 
        unique_labels = get_all_unique_labels(df['ner_char_labels_pylighter'])
    else: 
        unique_labels = get_all_unique_labels(df['ner_char_labels_pylighter']) + additional_label_names

    print(f'Unique labels found: {unique_labels}')
    additional_infos_cols = ['y', 'mnlp'] 
    additional_infos_cols = [col for col in additional_infos_cols if col in df]
    
    annotation = Annotation(
        df['txt'],
        additional_infos=df[additional_infos_cols] if additional_infos_cols != [] else None,
        labels_names=unique_labels
    )
    
    # load labels
    annotation.labels = df['ner_char_labels_pylighter'].tolist()
    
    return annotation


def remove_iob_preffix(labelname):
    return re.sub('^[IB]-', '', labelname)


def remove_iob_preffix_list(list_labels):
    return [remove_iob_preffix(l) for l in list_labels]


def get_all_unique_labels(l_all_labels_samples):
    from itertools import chain
    set_all_labels = set(list(chain.from_iterable(l_all_labels_samples)))
    set_labelnames = list(set([remove_iob_preffix(l) for l in set_all_labels if l not in 'O']))
    return set_labelnames


