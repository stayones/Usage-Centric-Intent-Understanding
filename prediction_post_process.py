import argparse
import json
import os
import random
import time

import gzip
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import openai
openai.api_key = "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"  # from Wendy
from scipy.stats import entropy
from sklearn import preprocessing
# from gpt_utils import wrap_prompt_chat, wrap_prompt_completion

def get_cate_property(cecc_count):
    cecc_count = pd.DataFrame.from_dict(cecc_count, orient="index")
    cecc_count = cecc_count.reset_index().rename(columns={"index": "ppt_cate", 0: "ppt_count"})
    cecc_count['bare_cate'] = [x.split("###")[0].strip() for x in cecc_count['ppt_cate']]
    new_df = cecc_count.groupby('bare_cate').agg(bare_count=('ppt_count', 'sum')).reset_index().rename(columns={"index": "bare_cate"})
    cecc_count =cecc_count.merge(new_df, on="bare_cate")
    # ppt percentage per raw_cate
    cecc_count['ppt_p'] = cecc_count['ppt_count'] / cecc_count['bare_count']
    return cecc_count.set_index('ppt_cate')

def mrr_calculate(prediction_rank, gold_ceccs):
    """
    param prediction_rank: dict of the predicted ceccs and rank
    param gold_ceccs: list of gold ceccs
    """
    cecc_max_reciprocal_rank, cecc_mean_rr = 0, 0
    v2c_hit_flag = False
    for g_cecc in gold_ceccs:
        if g_cecc in prediction_rank:
            # curr_g_rank = prediction_rank[g_cecc]
            curr_g_rank = prediction_rank.get(g_cecc)
            cecc_mean_rr += 1.0/curr_g_rank
            cecc_max_reciprocal_rank = max(cecc_max_reciprocal_rank, 1.0/curr_g_rank)
            v2c_hit_flag = True
    cecc_mean_rr /= len(gold_ceccs)
    return cecc_max_reciprocal_rank, cecc_mean_rr, v2c_hit_flag


def sample_change_ppt(prediction_rank, cate_info):
    used_ppt_cate = set()
    new_prediction_rank = {}
    unchange_count = 0
    raw_cate_index = {}
    # print(f"The number of original ranking is {len(prediction_rank)}")
    for i, (pce, r) in enumerate(prediction_rank.items()):
        if pce in cate_info.index:
            pce_bare = cate_info.loc[pce]['bare_cate']
            all_ppt_cates = cate_info[cate_info['bare_cate'] == pce_bare].sort_values('ppt_p', ascending=False)
            if pce_bare in raw_cate_index:
                now_index = raw_cate_index[pce_bare] + 1
            else:
                now_index = 0

            raw_cate_index.update({pce_bare: now_index})
            # print(f"Original length is {len(all_ppt_cates)}")
            # print(f"The rank of origianl prediction is {list(all_ppt_cates.index).index(pce)}")
            # new_ppt_cate = random.choices(list(all_ppt_cates.index), weights=all_ppt_cates['ppt_p'].to_numpy())[0]
            new_ppt_cate = list(all_ppt_cates.index)[now_index]
            if new_ppt_cate == pce:
                unchange_count += 1
            while new_ppt_cate in used_ppt_cate:
                print(now_index)
                print(raw_cate_index)
                print(pce_bare)
                print(list(all_ppt_cates.index)[now_index])
                print(list(all_ppt_cates.index)[now_index-1])
                exit("Not suppposed to be used")
                # all_ppt_cates = all_ppt_cates.drop(new_ppt_cate, axis=0)
                # all_ppt_cates['ppt_p'] /= sum(all_ppt_cates['ppt_p'])
                # print(f"New length is {len(all_ppt_cates)}")
                # print(f"The length of cate info is {cate_info.shape[0]}")
                try:
                    new_ppt_cate = random.choices(list(all_ppt_cates.index), weights=(all_ppt_cates['ppt_p'].to_numpy()))[0]
                except ValueError as e:
                    print(1.00001-all_ppt_cates['ppt_p'].to_numpy())
                    exit()
            used_ppt_cate.add(new_ppt_cate)
            new_prediction_rank[new_ppt_cate] = r

        else:
            print("Not found in the cecc set!!")
            new_prediction_rank[pce] = r
    # print(f"The length of the prediction is {len(prediction_rank)}")
    print(f"The number of the unchanged prediction {unchange_count}")
    # print(f"The lenght of unique is {len(unique_pre_rank)}")

    assert len(new_prediction_rank) == len(prediction_rank)
    # exit()
    return new_prediction_rank

def build_cate_rank_dict(dict_to_sort):
    return {p[0]: i + 1 for i, p in enumerate(sorted(dict_to_sort.items(), key=lambda x: x[1], reverse=True))}

def subsititute_ppt(prediction_file_path, cecc_path):
    with open(cecc_path) as f:
        cecc_count_dict = json.load(f)

    all_cate_info = get_cate_property(cecc_count_dict)
    cecc_ratio = {k: v/sum(cecc_count_dict.values()) for i, (k, v) in enumerate(cecc_count_dict.items())}
    total_max_mrr, total_mean_mrr = [], []
    miss_hit_count = 0
    start = time.time()
    with gzip.open(prediction_file_path,"rt", encoding="utf-8") as rfp:
        for idx, line in enumerate(rfp):
            prediction_entry = json.loads(line)
            gold_ceccs = prediction_entry['ceccs']
            predicted_ceccs = prediction_entry['predicted_ceccs']['weed']   # a dictinary of {cate: edge weight}
            if len(predicted_ceccs) == 0:
                predicted_ceccs = cecc_ratio
            predict_cecc_rank = build_cate_rank_dict(predicted_ceccs) # would return a dict as {cate_name: rank (start with 1)}
            assert max(predict_cecc_rank.values()) == len(predict_cecc_rank)
    
            # max_mrr_pre, mean_mrr, hit_lable_pre = mrr_calculate(prediction_rank=predict_cecc_rank, gold_ceccs=gold_ceccs)
            sub_cecc_rank = sample_change_ppt(prediction_rank=predict_cecc_rank, cate_info=all_cate_info)

            example_count = 5
            max_mrr, mean_mrr, hit_lable = mrr_calculate(prediction_rank=sub_cecc_rank, gold_ceccs=gold_ceccs)
            # if hit_lable_pre and hit_lable and 1/max_mrr_pre < 1/max_mrr:
            #     rank_to_predict = list(predict_cecc_rank.keys())
            #     prev_high_rank, sub_high_rank = int(1/max_mrr_pre), int(1/max_mrr)
            #     rank_to_sub = list(sub_cecc_rank.keys())
            #     print(f"Original highest cecc is {prev_high_rank}")
            #     for i in rank_to_predict[:prev_high_rank]:
            #         print(i)
            #     print(f"Now highest cecc is {sub_high_rank}")
            #     for i in rank_to_sub[:sub_high_rank]:
            #         print(i)
            #     print('============================')
            #     example_count -= 1
            #     if example_count < 0:
            #         exit()
                # print("The sub is not bad")
            total_mean_mrr.append(mean_mrr)
            total_max_mrr.append(max_mrr)
            if not hit_lable:
                miss_hit_count += 1
            if idx % 100 == 0:
                print(f"Now evaluating the {idx} entries")
                print(f"Total time cost {(time.time() - start)// 60} minutes")
                print(f"Now the avg mean mrr is {sum(total_mean_mrr)/ len(total_mean_mrr)}")
                print(f"Now the avg max mrr is {sum(total_max_mrr)/ len(total_max_mrr)}")
    print(f"After change the ppts:")
    print(f"The total miss hit is {miss_hit_count}")
    print(f"Now the avg mean mrr is {sum(total_mean_mrr)/ len(total_mean_mrr)}")
    print(f"Now the avg max mrr is {sum(total_max_mrr)/ len(total_max_mrr)}")

def subsitute_vague_to_concrete(vague_to_concrete, cecc_path):
    with open(cecc_path) as f:
        cecc_count_dict = json.load(f)

    all_cate_info = get_cate_property(cecc_count_dict)

    reseult_intent = []
    start = time.time()
    with open(vague_to_concrete) as f:
        for idx, line in enumerate(f):
            new_intent = {}
            intent = json.loads(line)
            new_intent['vague'] = intent['vague']
            new_oedges = sample_change_ppt(intent['oedges'], all_cate_info)
            new_intent['oedges'] = new_oedges
            if idx and idx % 100 == 0:
                print(f"Now changing the number {idx} intent...")
                print(f"Total time cost: {(time.time()-start)//3600} minutes...")

            reseult_intent.append(new_intent)
    with open(os.path.join(os.path.dirname(vague_to_concrete), "new_" + os.path.basename(vague_to_concrete)), "w") as f:
        for ir in reseult_intent:
            f.write(json.dumps(ir) + "\n")
    print("writing complete")
def cost_calculate_gpt(name, llm_info):
    """
    name -- str: The name of the model used
    token_number -- int: the number of token used (both prompt and completion)

    return -- float: the cost
    """
    # names can be changed later
    gpt_price = {
        "gpt-3.5-turbo-1106": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "text-davinci-003": 0.1200,
        "text-curie-001": 0.0120,
        "text-babbage-001": 0.0006,
        "text-ada-001": 0.0004,
    }
    if name in gpt_price and isinstance(gpt_price[name], dict):
        return (
            llm_info["usage"]["prompt_tokens"] / 1000 * gpt_price[name]["input"]
            + llm_info["usage"]["completion_tokens"] / 1000 * gpt_price[name]["output"]
        )
    elif name in gpt_price:
        return llm_info["usage"]["total_tokens"] / 1000 * (gpt_price[name])
    else:
        return 0.0

def cate_level_eval(prediction_file):
    total_max_mrr = []
    all_predictions = []
    with gzip.open(prediction_file, "rt", encoding="utf-8") as rfp:
        for idx, line in enumerate(rfp):
            all_predictions.append(json.loads(line))
    print(len(all_predictions))
    random.seed(24)
    all_predictions = random.sample(all_predictions, 1000)

    for idx, prediction in enumerate(all_predictions):
        # prediction_entry = json.loads(line)
        # cate_cecc_gold = [cc.split("###")[0] for cc in prediction_entry['ceccs']]
        cate_cecc_gold = prediction['ceccs']
        predicted_ceccs = list(prediction['predicted_ceccs']['weed'].keys())[:10]
        predictions = build_numerate_string(predicted_ceccs)
        ground_truth = build_numerate_string(cate_cecc_gold)
        print(f"The predictions\n{predictions}\n=*20")
        print(f"The gold\n{ground_truth}\n=*20")
        # predicted_ceccs = [cc.split("###")[0] for cc in prediction_entry['predicted_ceccs']['weed']]
        # max_mrr, mean_mrr, hit = mrr_calculate(prediction_rank=predicted_ceccs, gold_ceccs=cate_cecc_gold)
        validation_results= GPT4_eval(predictions, ground_truth)
        prediction.update({"gpt4_eval": validation_results['validation_list'], "max_rr": validation_results['max_rr']})
        total_max_mrr.append(validation_results['max_rr'])
    print(f"The final mean max mrr is {np.mean(total_max_mrr)}")
    with open(os.path.join(os.path.dirname(prediction_file), "gpt4_eval_" + os.path.basename(prediction_file).strip(".gz"))) as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print("Writing Complete!")

def postprocess(re_string):
    re_string = re_string.split(":")[-1]
    re_list = re_string.split("\n")
    if len(re_list) > 30:
        print(f"Wrong generation results\n{re_string}")
    result = []
    for r in re_list:
        if r and r[0].isdigit() and len(r) > 2:
            if r[1] == '.':
                result.append(r[2:].strip("\"").strip("\'").strip("\n").strip())
            else:
                result.append(r[1:].strip("\"").strip("\'").strip("\n").strip())
        elif len(r) > 2:
            result.append(r)
    return result

def build_numerate_string(content_list):
    res_string = ''
    for i in range(len(content_list)):
        res_string += f"{str(i)}. {content_list[i]}\n"
    return res_string

def GPT4_eval(prediction, ground_truth):
    GPT_ALIGN_PROMT = """
Here is a list of predicted categories:
{prediction}
Validate each prediction based on the ground truth categories[T/F].
Each prediction can be considered true when it is similar to one of the ground truth categories.
Ground truth categories:
{cecc}"""
    message = GPT_ALIGN_PROMT.format(
        prediction=prediction,
        cecc=ground_truth
    )
    model_dict = wrap_prompt_chat(message, "gpt-4", max_tokens=256)
    try:
        response = openai.ChatCompletion.create(**model_dict)
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(5)
        return None
    if response is not None:
        ret_text = response['choices'][0]['message']['content']
        # print(ret_text)
        cost = cost_calculate_gpt(model_dict['model'], response)
        validation_result = postprocess(ret_text)
        max_rr = 1 / 126520
        for v_i in range(len(validation_result)):
            if validation_result[v_i].lower().startswith("t"):
                max_rr = max(max_rr, 1 / (v_i + 1))
                break
        print(f"The max_mrr is {max_rr}")
        print(ret_text)
        return {"max_rr": max_rr, "cost": cost, "validation_list": validation_result}


def chat_gpt_end_to_end(prediction_file):
    GPT_PREDICT_PROMPT="""
    Intents: 
    {intents}
    Given the intents, please predict the top 10 kinds of products that will be useful for these intents.
    A kind of product is the concatenation of a fine-grained category from the Amazon Review Dataset and a useful property. For example: Clothing, Shoes & Jewelry|Men|Watches|Wrist Watches ### leather.
    Kinds of products:
    1. """

    GPT_ALIGN_PROMT_trubo = """
Here is a list of predicted categories:
{prediction}
Ground truth categories:
{cecc}
Validate each predicted category based on the ground truth category[T/F].
Each prediction can be considered true when it is similar to one of the ground truth categories.
        """
    cost = 0
    gpt_results = []
    all_predictions = []
    with gzip.open(prediction_file, "rt", encoding="utf-8") as rfp:
        for idx, line in enumerate(rfp):
            all_predictions.append(json.loads(line))
    print(len(all_predictions))
    random.seed(24)
    all_predictions = random.sample(all_predictions, 1000)
    print(len(all_predictions))
    for prediction_entry in all_predictions:
        input_intents = prediction_entry['intents']
        intent_string = build_numerate_string([i[1] for i in input_intents])

        message = GPT_PREDICT_PROMPT.format(intents=intent_string)
        # print(message)
        model_dict = wrap_prompt_chat(message, "gpt-3.5-turbo-1106", max_tokens=256)
        try:
            response = openai.ChatCompletion.create(**model_dict)
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        if response is not None:
            if idx and idx % 100 ==0:
                print(f"Now is the {idx}")
            ret_text = response['choices'][0]['message']['content']
            cost += cost_calculate_gpt(model_dict['model'], response)
            results = postprocess(ret_text)

            gpt_results.append({
                "asin": prediction_entry['asin'],
                "ceccs": prediction_entry['ceccs'],
                "gpt_predictions": results
            })
            print(ret_text)
            print(f"Now the cost is {cost}")
            result_list = build_numerate_string(results)
            ground_truth_list = build_numerate_string(prediction_entry['ceccs'])
            print(f"Now the ground truth are:\n{ground_truth_list}==================")
            valid_result = GPT4_eval(result_list, ground_truth_list)
            if valid_result:
                gpt_results[-1].update({"max_rr": 0})
                continue

            gpt_results[-1].update({"max_rr": valid_result['max_rr']})
            cost += valid_result['cost']

    with open(os.path.join(os.path.dirname(prediction_file), "gpt4_prediction_" + os.path.basename(prediction_file).strip(".gz")), "w") as fw:
        for r in gpt_results:
            fw.write(json.dumps(r) + "\n")
    print(f"There are {len(gpt_results)} saved!")


def round_trip_eval(prediction_file):
    total_max_mrr, total_mean_mrr = [], []
    start = time.time()
    miss_hit = 0
    total_co_buy, totol_other_co_buy = 0, 0
    with gzip.open(prediction_file,"rt", encoding="utf-8") as rfp:
        for idx, line in enumerate(rfp):
            prediction_entry = json.loads(line)
            cobuy_cecc = prediction_entry['co_buy_ceccs']
            cate_cecc = [cc.split("###")[0] for cc in prediction_entry['ceccs']]
            cate_cecc = set(cate_cecc)
            if len(cobuy_cecc) > 0:
                total_co_buy += 1
                other_cate_cobuy_cecc = [cc for cc in cobuy_cecc if cc.split('###')[0] not in cate_cecc]
                if len(other_cate_cobuy_cecc) > 0:
                    totol_other_co_buy += 1
                    predicted_co_buy_cecc = prediction_entry['predicted_cobuy_ceccs']['weed']
                    predicted_co_buy_rank = build_cate_rank_dict(predicted_co_buy_cecc)
                    max_mrr, mean_mrr, hit = mrr_calculate(prediction_rank=predicted_co_buy_rank, gold_ceccs=other_cate_cobuy_cecc)
                    if not hit:
                        miss_hit += 1
                    total_max_mrr.append(max_mrr)
                    total_mean_mrr.append(mean_mrr)
            if idx and idx % 50 == 0:
                print(f"Now evaluating the {idx} entries")
                print(f"Total time cost {(time.time() - start)// 60} minutes")
                print(f"Now the avg mean mrr is {sum(total_mean_mrr)/ len(total_mean_mrr)}")
                print(f"Now the avg max mrr is {sum(total_max_mrr)/ len(total_max_mrr)}")
    print("==========================================")
    print(f"The round trip prediction results are:")
    print(f"There is total {total_co_buy} co-buy evaluate entries and {totol_other_co_buy} other cobuy")
    print(f"When ther is other co-buy but miss hit is {miss_hit}")
    print(f"Now the avg mean mrr is {sum(total_mean_mrr)/ len(total_mean_mrr)}")
    print(f"Now the avg max mrr is {sum(total_max_mrr)/ len(total_max_mrr)}")

def intent_to_cate_entropy(intent_to_cate_file):
    entropy_result = {}
    non_empty_intent, example = 0, 10
    with open(intent_to_cate_file) as f:
        for idx, line in enumerate(f):
            vague_to_cecc = json.loads(line)
            if idx and idx % 50 == 0:
                print(f"Now evaluating {idx}")
                print(f"Average entopy now {sum(entropy_result.values())/ len(entropy_result)}")
            bare_cate_dict = vague_to_cecc['oedges']
            # for i, (k, v) in enumerate(vague_to_cecc['oedges'].items()):
            #     if len(k.split("###")) == 1:
            #         bare_cate_dict[k] = v
            if len(bare_cate_dict) > 0:
                non_empty_intent += 1
                ceccs_scores = pd.DataFrame.from_dict(bare_cate_dict, orient="index")
                entropy_weed = entropy(preprocessing.normalize([np.array(ceccs_scores.loc[:,'weed'])])[0])
                entropy_result.update({vague_to_cecc['vague']: entropy_weed})
                # if entropy_weed < 0.7 and entropy_weed > 0.6:
                if entropy_weed < 1e-4:
                    example -= 1
                    print(f"Entropy {entropy_weed}, the intent is \n {vague_to_cecc['vague']}")
                    print(np.array(ceccs_scores.loc[:,'weed']))
            if example < 0:
                break

    entropy_path = os.path.join(os.path.dirname(intent_to_cate_file), "entropy_test_" + os.path.basename(intent_to_cate_file))
    print(f"Saving entropy results to {entropy_path}")
    with open(entropy_path, "w") as f:
        f.write(json.dumps(entropy_result))

def drawing_entropy_graph(elec_entropy_f, clothing_entropy_f):
    with open(clothing_entropy_f) as f:
        clothing_entropy = json.load(f)
    with open(elec_entropy_f) as f:
        elec_entropy = json.load(f)

    clothing_entropy_values = list(clothing_entropy.values())
    print(np.average(clothing_entropy_values))
    
    elec_entropy_values = list(elec_entropy.values())
    print(np.average(elec_entropy_values))
    his_bins = np.append(np.arange(0,3, 1/50), [3, int(max(clothing_entropy_values + elec_entropy_values)) + 1])
    his_index = np.append(np.arange(0,3, 1/5), [3])
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    plt.subplots_adjust(hspace=0.3)
    # from matplotlib.ticker import FixedLocator
    
    N, bins = np.histogram(clothing_entropy_values,weights=np.ones(len(clothing_entropy_values))/ len(clothing_entropy_values), bins=his_bins)
    N_e, bins_e = np.histogram(elec_entropy_values,weights=np.ones(len(elec_entropy_values))/ len(elec_entropy_values),bins=his_bins)
    # N, bins,patches = ax1.hist(clothing_entropy_values,weights=np.ones(len(clothing_entropy_values))/ len(clothing_entropy_values), bins=20, label="Clothing")
    # N1, bin1, patch = ax2.hist(elec_entropy_values,weights=np.ones(len(elec_entropy_values))/ len(elec_entropy_values),bins=20, label="Electronics")
    # locator = FixedLocator(np.arange(len(bins)))
    # ax1.hist(bins[:-1], bins=bins,weights=N)
    ax1.title.set_text('Entropy of Intents in Clothing')
    ax2.title.set_text('Entropy of Intents in Electronics')
    # ax1.tick_params(bottom = False, labelbottom = False) 
    # print(his_index)
    his_index = [round(i, 2) for i in his_index]
    l = [str(round(v, 2)) if round(v, 2) in his_index else "" for v in his_bins[:-1]]
    # print(l)
    ax1.set_xticks(his_bins[:-1], labels=l)
    ax2.set_xticks(his_bins[:-1], labels=[str(round(v, 2)) if round(v, 2) in his_index else "" for v in his_bins[:-1]])
    # ax2.set_xticks(his_bins)
    f1 = ax1.bar(bins[:-1], N, width=0.02, align='edge', color=(0.3, 0.8, 0.2, 0.6))
    
    ax1.bar_label(f1, [f"{round(i*100)}%" if i > 0.01 else "" for i in N] )
    f2 = ax2.bar(bins_e[:-1], N_e, width=0.02, align='edge', color=(0.2, 0.3, 0.8, 0.6))
    ax2.bar_label(f2, [f"{round(i*100)}%" if i > 0.01 else "" for i in N_e])
    
    

    ax2.set_ylim(top=1)
    ax2.set_yticklabels([f'{x:.0%}' for x in ax2.get_yticks()])
    ax1.set_ylim(top=1)
    ax1.set_yticklabels([f'{x:.0%}' for x in ax1.get_yticks()])
    # fracs = ((N**(1/2))/ N.max())
    # fracs1 = ((N1**(1/2))/ N1.max())
    # norm = colors.Normalize(fracs.min(), fracs.max())
    # norm1 = colors.Normalize(fracs1.min(), fracs1.max())
    # for thisfrac, thispatch in zip(fracs, patches):
        # color = plt.cm.viridis_r(norm(thisfrac))
    #     thispatch.set_facecolor(color)
    # for thisfrac, thispatch in zip(fracs1, patch):
    #     color = plt.cm.magma_r(norm1(thisfrac))
    #     thispatch.set_facecolor(color)
    plt.savefig('folkscope/images/intent_entropy_test.pdf')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="", help="The file path to the prediction path")
    parser.add_argument("--ceccs", default="", help="The file path to the cecc count file")
    parser.add_argument("--intent", default="", help="The file path to the intent to cecc file")
    parser.add_argument("--task", default="", required=True, help="The post prediction task")
    args = parser.parse_args()

    if args.task == "sub_ppt":
        subsititute_ppt(args.prediction, args.ceccs)
    elif args.task == "cross_cate":
        round_trip_eval(args.prediction)
    # elif args.task == "chatgpt":
    #     chat_gpt_end_to_end(args.prediction)
    elif args.task == "re_eval":
        cate_level_eval(args.prediction)

    # subsitute_vague_to_concrete(args.prediction, args.ceccs)
    # round_trip_eval(args.prediction)
    # intent_to_cate_entropy(args.intent)
    # cate_level_eval(args.prediction)
    # drawing_entropy_graph('folkscope/raw_cate/elec_populated_graphs/OWMcount/local/entropy_vague2concrete_local.OWMcount.json',
    #                       'folkscope/raw_cate/clothing_populated_graphs/OWMcount/local/entropy_vague2concrete_local.OWMcount.json')
