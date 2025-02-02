import clayrs.content_analyzer as ca
import os
import pandas as pd
import clayrs.evaluation as eva
import gc


dataset = 'lastfm'

domain = {
    'lastfm': 'music',
    'dbbook': 'book',
    'movielens': 'movie',
}[dataset]

relevant_threshold=1
ks = [1, 3, 5, 10]
predicted_score_path = f"DataPreprocessing/data/{dataset}"
models_names = [f'LLaMA_{domain.capitalize()}_graph', \
                f'LLaMA_{domain.capitalize()}_no_kn', f'LLaMA_{domain.capitalize()}_Text', \
                f'LLaMA_{domain.capitalize()}_graph_text', f'LLaMA_{domain.capitalize()}_collaborative',\
                f'LLaMA_{domain.capitalize()}_graph_text_double', f'LLaMA_{domain.capitalize()}_collaborative_graph',\
                f'LLaMA_{domain.capitalize()}_collaborative_text', f'LLaMA_{domain.capitalize()}_collaborative_graph_text'] 
csv_truth_1 = ca.CSVFile(os.path.join(predicted_score_path, "ground_truth.tsv"), separator="\t")
original_rating_path = f'MetricCalculation/datasets/{dataset}'
csv_orignal_ratings_1 = ca.CSVFile(os.path.join(original_rating_path, "original_ratings.tsv"), separator="\t")
original_ratings = ca.Ratings(csv_orignal_ratings_1)
dict_results = {}
results_list=[]
for baseline in models_names:
    if not os.path.exists(os.path.join(predicted_score_path, f"predicted_score_{baseline}.tsv")):
        print(f"{baseline} not found")
        continue
    metric_list = []
    for k in ks:
        metric_list.extend([
            eva.FMeasureAtK(k=k, relevant_threshold=relevant_threshold),
            eva.PrecisionAtK(k=k, relevant_threshold=relevant_threshold), # 
            eva.RecallAtK(k=k, relevant_threshold=relevant_threshold),
            eva.NDCGAtK(k=k),
            eva.AvgPopularityAtK(k=k),
        ])
    metric_list.extend([
            eva.FMeasure(relevant_threshold=relevant_threshold),
            eva.Precision(relevant_threshold=relevant_threshold), # 
            eva.Recall(relevant_threshold=relevant_threshold),
            eva.NDCG(),
            eva.AvgPopularity(),
        ])
    csv_rank_1 = ca.CSVFile(os.path.join(predicted_score_path, f"predicted_score_{baseline}.tsv"), separator="\t")

    rank_1 = ca.Rank(csv_rank_1)
    truth_1 = ca.Ratings(csv_truth_1)

    rank_list = [rank_1]
    truth_list = [truth_1]
    import clayrs.evaluation as eva

    em = eva.EvalModel(
        pred_list=rank_list,
        truth_list=truth_list,
        metric_list=metric_list,
        original_ratings =original_ratings
    )
    sys_result, users_result =  em.fit()
    sys_result = sys_result.loc[['sys - mean']]
    sys_result.reset_index(drop=True, inplace=True)
    sys_result['model'] = baseline
    sys_result.columns = [x.replace(" - macro", "") for x in sys_result.columns]
    results_mean = users_result.fillna(0).mean()
    results_mean.index = [x.replace(" - macro", "") for x in results_mean.index]
    dict_results[baseline] = sys_result
    results_mean['model'] = baseline
    results_list.append(results_mean)
    pd.to_pickle(dict_results, os.path.join(predicted_score_path, f'results_{dataset}.pkl'))
    gc.collect()
results = pd.concat([v for v in dict_results.values()]).reset_index(drop=True)
results.to_csv(os.path.join(predicted_score_path, f'results_{dataset}.csv'), index=False, sep=';')