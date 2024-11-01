# Usage-Centric-Intent-Understanding

This is the repository for the **Product Recovery Benchmark** proposed in the paper _"A Usage-centric Take on Intent Understanding in E-Commerce"_ accepted by EMNLP 2024.

## Example Dataset (Refactored FolkScope)

- Clothing KG ([here](/Folkscope_re/Clothing))
- Electronic KG ([here](/Folkscope_re/Electronic))

## Evaluation Framework

To run our evaluation script and(or) the statistic experiments

### Clone project

Execute the following command in the root directory:

```Bash
git clone https://github.com/stayones/Usgae-Centric-Intent-Understanding.git

cd Usgae-Centric-Intent-Understanding
```

### Setup enviornment

```Bash
conda create -n uiecc

conda activate uiecc

pip install -r requirements.txt
```

If you encounter any issue for building wheels with `tokenizers`, please see possible solutions [here](https://stackoverflow.com/questions/77265938/cargo-rustc-failed-with-code-101-could-not-build-wheels-for-tokenizers-which).

### Evaluation

To be updated

### Statistic experiment

After obtaining the evaluation results on the test set (see example prediction result in /prediction_result folder), we conduct the following experiments to further inspect the quality of intents mined by FolkScope.

#### LM Re-ranking (GPT)

```Bash
python -u evaluate_folkscope.py --task lm_rerank --occ_weight_mode [count/pmi] --score_reduction sum --smooth_ratio 0.01 --eval_subset [dev/test/all] --rerank_scrkey weed --rerank_n 10 --rerank_model_type gpt --rerank_model_name gpt-3.5-turbo-0613 (--do_round_trip) --cate_name [clothing_populated/elec_populated] 
```

- the `--occ_weight_mode` here is only used to choose the type of graph to be augmented.
- **Note**: include your own openai api key before running this experiment.

#### Property corruption experiment

```Bash
python -u prediction_post_process.py --task sub_ppt --prediction path/to/evaluation_result --ceccs path/to/cecc_cnt
```

#### Cross category co-buy kinds of products evaluation experiments

 ```Bash
 python -u prediction_post_process.py --task cross_cate --prediction path/to/evaluation_result
 ```

#### Gpt-turbo-1106 predict kinds of products and gpt-4 evaluation experiment

```Bash
python -u prediction_post_process.py --task chatgpt --prediction path/to/evaluation_result
```

#### Do gpt-4 re-evaluation experiment

```
python -u prediction_post_process.py --task re_eval --prediction path/to/evaluation_result
```
