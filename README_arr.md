# Scripts to run for the FolkScope analysis
The full dataset will be released after acceptance, we exclude the Clothing domain since it exceeds the upload limit.

### Analysis experiment
After obtaining the evaluation results on the test set (see example prediction result in /prediction_result folder), we conduct the following experiments to further inspect the quality of intents mined by FolkScope.
- Do LM Re-ranking (GPT): ` python -u evaluate_folkscope.py --task lm_rerank --occ_weight_mode [count/pmi] --score_reduction sum --smooth_ratio 0.01 --eval_subset [dev/test/all] --rerank_scrkey weed --rerank_n 10 --rerank_model_type gpt --rerank_model_name gpt-3.5-turbo-0613 (--do_round_trip) --cate_name [clothing_populated/elec_populated] ` (the `--occ_weight_mode` here is not directly used, but it is used to determine the version of graph to be augmented.). 
  - Please remember to include your own openai api key before running this experiment.
- Do property corruption experiment: `python -u prediction_post_process.py --task sub_ppt --prediction path/to/evaluation_result --ceccs path/to/cecc_cnt `
- Do cross category co-buy kinds of products evaluation experiments: `python -u prediction_post_process.py --task cross_cate --prediction path/to/evaluation_result`
- Do gpt-turbo-1106 predict kinds of products and gpt-4 evaluation experiment: `python -u prediction_post_process.py --task chatgpt --prediction path/to/evaluation_result`
- Do gpt-4 re-evaluation experiment: `python -u prediction_post_process.py --task re_eval --prediction path/to/evaluation_result` 
