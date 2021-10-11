import pandas as pd
from tqdm.notebook import tqdm
from fuzzywuzzy import fuzz

def make_query_df_db_and_rank(df_db, query_txt, cols_db_to_show_in_match_results):
    '''
    query_txt = ner_predictions['text'], where `ner_predictions` is from row['sentences_ner_final_predictions_chained']
        or it can also be normal text
    cols_db_to_show_in_match_results = COLS_USED_TO_QUERY_DB_OFI
    '''
    existing_scores_cols = [col for col in df_db.columns if col.startswith('score')]
    cols_to_drop = existing_scores_cols
    cols_to_drop = [col for col in cols_to_drop if col in df_db]
    df_db.drop(cols_to_drop, axis=1, inplace=True)
    
    df_db['score'] = df_db['query'].apply(lambda txt: fuzz.token_set_ratio(query_txt, txt))
    
    df_db = df_db.sort_values(by='score', ascending=False) # Higher score means better match

    return {
        'df_db': df_db,
        'd_df_db_top10': df_db[cols_db_to_show_in_match_results + ['score']].iloc[:10].to_dict()
    }


def get_best_result_from_db_rank(d_ret_db_rank):
    df_db = d_ret_db_rank['df_db']

    best_db_row = df_db.iloc[0]
    best_db_idx = best_db_row.name
    d_result = {}
    d_result['index_best_match'] = best_db_idx
#         d_result['matched_query_in_df_db'] = best_db_row.query # May lead to confusion to whoever reads it. Will not give this info for now
    d_result['d_df_db_top10'] = d_ret_db_rank['d_df_db_top10']
    d_result['dict_best_match'] = pd.DataFrame(d_ret_db_rank['d_df_db_top10']).iloc[0].to_dict() # .iloc[0] is best match, as it will be sorted by score

    return d_result



def set_matches_df_db(df, df_db, cols_db_to_show_in_match_results):
    list_d_results = [] # To be added to df
    
    for i, row in tqdm(df.iterrows(), total=len(df)): # For each row (mail)
        list_this_doc_ner_matches_results = []
        for list_ner_predictions_found_in_sentence in row['sentences_ner_final_predictions']: # For each sentence
            list_matches_in_sentence = []
            for ner_prediction in list_ner_predictions_found_in_sentence: # For each NER finding
                # Get results of queries against db 
                d_ret_db_rank = make_query_df_db_and_rank(
                    df_db=df_db,
                    query_txt=ner_prediction['text'],
                    cols_db_to_show_in_match_results=cols_db_to_show_in_match_results
                )
                d_match_results = get_best_result_from_db_rank(d_ret_db_rank) # get results in a dict. Best index is in here
                list_matches_in_sentence.append(d_match_results)
            list_this_doc_ner_matches_results.append(list_matches_in_sentence)
        list_d_results.append(list_this_doc_ner_matches_results) # one element per mail
    df['ner_matching_with_db_results'] = list_d_results
#     df['index_best_match_db'] = df['ner_matching_results'].apply(lambda d: d['index_best_match'])
    return df