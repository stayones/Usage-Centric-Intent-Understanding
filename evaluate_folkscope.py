import json
import gzip
from locale import currency
import os
import argparse
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import sys
import time
import math
from typing import Union
from transformers import AutoModel, AutoTokenizer
import torch
import openai
from gpt_utils import wrap_prompt_chat, wrap_prompt_completion, process_list_from_output

from build_ecc_graphs_fksc import load_source_dict, load_asin2asin
from evaluate_ecc_graphs_utils import load_gz_jsonl, calc_recrank, load_graphs, kl_divergence， rmse

openai.api_key = "Your api key"
CHAT_MODELS = ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
MAX_NUM_RETRIES = 5




def aggr_likelihood(gold, prediction, majorities, smooth_ratio, keys_size):
    """
    Takes the average likelihood over all intents in gold.
    """
    pred_aggr_l = 0.0
    majority_aggr_l = 0.0

    background_noise_scr = smooth_ratio / keys_size
    rand_aggr_l = 1.0 / keys_size
    for gk in gold:
        if gk in prediction:
            pred_aggr_l += prediction[gk]*(1-smooth_ratio)
        else:
            pred_aggr_l += background_noise_scr
        if gk in majorities:
            majority_aggr_l += majorities[gk]*(1-smooth_ratio)
        else:
            majority_aggr_l += background_noise_scr
    pred_aggr_l /= len(gold)
    majority_aggr_l /= len(gold)
    return pred_aggr_l, rand_aggr_l, majority_aggr_l


def aggr_rr(gold, prediction, majorities_ranks, smooth_ratio, keys_size):
    """
    Takes the average likelihood over all intents in gold.
    majorities_ranks are already sorted.

    """
    pred_rrs = []
    majority_rrs = []

    prediction = {k: v for (k, v) in sorted(prediction.items(), key=lambda item: item[1], reverse=True)}
    prediction = {k: i for i, k in enumerate(prediction.keys())}

    for gk in gold:
        if gk in prediction:
            pred_rrs.append(1/(prediction[gk]+1))
        else:
            pred_rrs.append(0)
        if gk in majorities_ranks:
            majority_rrs.append(1/(majorities_ranks[gk][1]+1))
        else:
            majority_rrs.append(0)
    pred_rr = sum(pred_rrs) / len(gold)
    majority_rr = sum(majority_rrs) / len(gold)
    return pred_rr, majority_rr


def aggregate_correspondences(source_eccs, s2t_graph, reduce: str, entscr_key: str, normalize: bool, do_rank: bool, keep_size: Optional[int], graph_target: str,
                              skip_parent_ratio: float) -> Dict[str, float]:
    """
    Given a list of source ECCs, return the corresponding target ECCs.
    This implementation is a simple average of all the target ECCs that are predicted from the respective source ECC mentions.
    Note that the source ECCs have not been deduplicated here, so multiple matches of the same source ECC will have a more significant effect.
    source_eccs: a list of source ECCs
    s2t_graph: a dictionary mapping source ECCs to target ECCs
    reduce: method to reduce the lists of concrete ECCs from different sources into a single list. Options: 'sum', 'avg', 'max'
    """
    assert graph_target in ['v', 'c']
    t_eccs = {}
    t_ecc_instance_cnts = {}
    if isinstance(source_eccs, list):
        use_weights = False
    elif isinstance(source_eccs, dict):
        use_weights = True
    else:
        raise NotImplementedError

    for s_ecc in source_eccs:
        if use_weights:
            s_ecc_weight = source_eccs[s_ecc]
        else:
            s_ecc_weight = 1.0
        if s_ecc_weight == 0:
            s_ecc_weight = 1e-3
        if s_ecc in s2t_graph:
            skip_ts = set()
            if graph_target == 'c':
                for t_ecc in s2t_graph[s_ecc]:
                    t_ecc_split = t_ecc.split(' ### ')
                    if len(t_ecc_split) > 1:
                        parent = t_ecc_split[:-1]
                        parent = ' ### '.join(parent)
                        if parent in s2t_graph[s_ecc] and (s2t_graph[s_ecc][parent][entscr_key] == 0 or (s2t_graph[s_ecc][t_ecc][entscr_key] / s2t_graph[s_ecc][parent][entscr_key] > skip_parent_ratio)):
                            skip_ts.add(parent)
            for t_ecc in s2t_graph[s_ecc]:
                if t_ecc in skip_ts:
                    continue
                if t_ecc not in t_eccs:
                    t_eccs[t_ecc] = 0.0
                    t_ecc_instance_cnts[t_ecc] = 0
                t_ecc_instance_cnts[t_ecc] += 1
                if reduce in ['sum', 'avg']:  # all the other target ECCs not in the graph will be treated as 0.0 in the average
                    t_eccs[t_ecc] += s2t_graph[s_ecc][t_ecc][entscr_key] * s_ecc_weight
                elif reduce == 'max':
                    t_eccs[t_ecc] = max(t_eccs[t_ecc], s2t_graph[s_ecc][t_ecc][entscr_key]*s_ecc_weight)
                else:
                    raise ValueError(f"Unknown reduce method {reduce}")
    
    if reduce == 'avg':
        for t_ecc in t_eccs:
            t_eccs[t_ecc] /= t_ecc_instance_cnts[t_ecc]
    else:
        pass
    
    if normalize:
        total_sum = 0.00000000001  # avoid division by 0
        for t_ecc in t_eccs:
            total_sum += t_eccs[t_ecc]
        for t_ecc in t_eccs:
            t_eccs[t_ecc] /= total_sum
    
    if do_rank or keep_size is not None:
        t_eccs = {k: v for k, v in sorted(t_eccs.items(), key=lambda item: item[1], reverse=True)}
        if keep_size is not None:
            t_eccs = {k: v for k, v in list(t_eccs.items())[:keep_size]}
        else:
            pass
    else:
        pass
    
    return t_eccs


def aggregate_twohop_correspondences(source_eccs, s2s_graph, s2t_graph, reduce: str, entscr_key: str, normalize: bool, do_rank: bool,
                                     max_medium_eccs: int, keep_size: int, twohop_epsilon: float, graph_target: str, skip_parent_ratio: float) -> Dict[str, float]:
    t_eccs = {}  # aggregation of all the target ECCs from different sources are defaulted to SUM
    if reduce != 'sum':
        print(f"Warning: reduce method {reduce} is not supported for two-hop aggregation. Defaulting to sum.", file=sys.stderr)
    
    if isinstance(source_eccs, list):
        use_weights = False
    elif isinstance(source_eccs, dict):
        use_weights = True
    else:
        raise NotImplementedError
    
    for s_ecc in source_eccs:
        if use_weights:
            s_ecc_weight = source_eccs[s_ecc]
        else:
            s_ecc_weight = 1.0
        if s_ecc_weight == 0:
            s_ecc_weight = 1e-3
        if s_ecc not in s2s_graph:
            continue
        medium_eccs = s2s_graph[s_ecc]
        medium_eccs = sorted([(m, s2s_graph[s_ecc][m][entscr_key]*s_ecc_weight) for m in medium_eccs], key=lambda x: x[1], reverse=True)
        medium_eccs = {m[0]: m[1] for m in medium_eccs[:max_medium_eccs] if m[1] > twohop_epsilon}   

        # Since we are normalizing / ranking them in the end, we don't need to normalize them here.
        curr_s_twohop_teccs = aggregate_correspondences(medium_eccs, s2t_graph, reduce, entscr_key, normalize=False, do_rank=False, keep_size=None, graph_target=graph_target, skip_parent_ratio=skip_parent_ratio)
        for t_ecc in curr_s_twohop_teccs:
            if t_ecc not in t_eccs:
                t_eccs[t_ecc] = 0.0
            t_eccs[t_ecc] += curr_s_twohop_teccs[t_ecc]
    
    if normalize:
        total_sum = 0.00000000001
        for t_ecc in t_eccs:
            total_sum += t_eccs[t_ecc]
        for t_ecc in t_eccs:
            t_eccs[t_ecc] /= total_sum
    
    if do_rank or keep_size is not None:
        t_eccs = {k: v for k, v in sorted(t_eccs.items(), key=lambda item: item[1], reverse=True)}
        if keep_size is not None:
            t_eccs = {k: v for k, v in list(t_eccs.items())[:keep_size]}
        else:
            pass
    else:
        pass
    
    return t_eccs


def assemble_eval_data(source_path: str, c_path: str, assembled_opath: str, verbose: bool, quiet: bool):

    entries = []
    with open(source_path, 'r', encoding='utf8') as ifp:
        for lidx, line in enumerate(ifp):
            if lidx % 1000 == 0:
                print(f"Processed {lidx} lines.")
            if len(line) < 2:
                continue
            jobj = json.loads(line)
            entries.append(jobj)

    asin2cecc_dict, asin2intent_dict = load_source_dict(source_path=source_path, debug=verbose, entropy=True)  # cecc_equally_true option only affects cate_D_dict (not used), so we can set it to False here.
    if not quiet:
        print(f"CECC mappings loaded!")
    # These are needed from the original meta file for round-trip evaluation.
    asin2asin_dict = load_asin2asin(meta_path=c_path, view2buy_rel_weight=0.0, verbose=verbose)  # we only treat co-buy relations as gold labels.
    if not quiet:
        print(f"ASIN mappings loaded!")

    print('Found %d ASINs with all available types of ECCs' % len(asin2cecc_dict))

    if not quiet:
        print(f"Organizing ECC mappings...")

    if not quiet:
        print(f"Eval mappings ready!")

    print(f"Saving assembled eval data to {assembled_opath}")
    with open(assembled_opath, 'w', encoding='utf8') as afp:
        for asin in asin2cecc_dict:  # only for those asins with v-c corresbondances, we test for reconstruction of co-buy relations.
            assert isinstance(asin2cecc_dict[asin], list)
            assert asin in asin2intent_dict
            if asin not in asin2asin_dict:
                print(f"ASIN {asin} not found in asin2asin_dict!!!!!!!", file=sys.stderr)
                cobuy_cecc_list = []
            else:
                cobuy_offer_list = [x[0] for x in asin2asin_dict[asin]]
                cobuy_cecc_list = []
                for oth_off in cobuy_offer_list:
                    if oth_off in asin2cecc_dict:
                        cobuy_cecc_list += asin2cecc_dict[oth_off]
                cobuy_cecc_list = list(set([x for x in cobuy_cecc_list if x not in asin2cecc_dict[asin]]))  # remove the ceccs that are already in the ceccs of the asin itself, then deduplicate.
            out_item = {'asin': asin, 'ceccs': asin2cecc_dict[asin], 'intents': asin2intent_dict[asin], 'co_buy_ceccs': cobuy_cecc_list}
            out_line = json.dumps(out_item, ensure_ascii=False)
            afp.write(out_line + '\n')

    print(f"Eval data saved!")


def random_sized_sample(population: dict, size: int, portion: float, RHO: float = 0.5):
    if len(population) < size:
        return population

    rho = random.random()
    if rho < RHO:
        sample_ = list(population.items())[:size]
    else:
        sample_ = list(population.items())[:int(len(population) * portion)]

    sample = {k: v for k, v in sample_}
    return sample


def merge_two_predictions(predictions, weights):
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    aggr = {}
    for p, w in zip(predictions, weights):
        for k, v in p.items():
            if k not in aggr:
                aggr[k] = 0.0
            aggr[k] += v * w
    
    return aggr


def auto_eval_main(assembled_opath, graph_opath_generic: str, cecc2cnt_path: str, 
                   vecc2cnt_path: str, score_reduction: str, smooth_ratio: float, max_medium_eccs: int, scores_opath: str, predictions_opath: str, 
                   do_round_trip_eval: bool, debug: bool, verbose: bool, quiet: bool, keep_size: int, twohop_weight: float,
                   twohop_epsilon: float, max_veccs_per_asin: int, skip_parent_ratio: float, label_maxi_specificity: bool,
                   no_disk_cache: bool):

    # first load the vague_ecc <-> concrete_ecc correspondences
    try:
        with open(assembled_opath, 'r', encoding='utf8') as afp:
            eval_entries = []
            eligible_cobuy_cecc_dict = {}  # these are the asins that have both v-c correspondances and co-buy relations.
            for lidx, line in enumerate(afp):
                jobj = json.loads(line)
                eval_entries.append(jobj)
                if len(jobj['co_buy_ceccs']) > 0:
                    eligible_cobuy_cecc_dict[jobj['asin']] = jobj['co_buy_ceccs']
        with open(cecc2cnt_path, 'r', encoding='utf8') as cfp:
            cecc2cnt_dict = json.load(cfp)
            total_cecc_instances_cnt = 0
            for cecc in cecc2cnt_dict:
                total_cecc_instances_cnt += cecc2cnt_dict[cecc]
            cecc2ratio_dict = {cecc: cecc2cnt_dict[cecc] / total_cecc_instances_cnt for cecc in cecc2cnt_dict}
            cecc2ratio_lst = sorted(cecc2ratio_dict.items(), key=lambda x: x[1], reverse=True)
            cecc2ratio_dict = {k: (v, i) for i, (k, v) in enumerate(cecc2ratio_lst)}
            del cecc2ratio_lst
        cecc_maj_rmse_preds = {k: v[0] for (k, v) in cecc2ratio_dict.items()}
        with open(vecc2cnt_path, "r", encoding='utf8') as vfp:
            vecc2cnt_dict = json.load(vfp)
            total_vecc_instances_cnt = 0
            for vecc in vecc2cnt_dict:
                total_vecc_instances_cnt += vecc2cnt_dict[vecc]
            vecc2ratio_dict = {vecc: vecc2cnt_dict[vecc] / total_vecc_instances_cnt for vecc in vecc2cnt_dict}
            vecc2ratio_lst = sorted(vecc2ratio_dict.items(), key=lambda x: x[1], reverse=True)
            vecc2ratio_dict = {k: (v, i) for i, (k, v) in enumerate(vecc2ratio_lst)}
            del vecc2ratio_lst
        vecc_maj_rmse_preds = {k: v[0] for (k, v) in vecc2ratio_dict.items()}

    except FileNotFoundError as e:
        print('No assembled eval data found at %s' % assembled_opath)
        return
    
    print(f"Loaded {len(eval_entries)} ASINs with v-c correspondances and {len(eligible_cobuy_cecc_dict)} ASINs with co-buy relations.")

    # Then load the graphs
    concrete2vague_dict, vague2concrete_dict, vague2vague_dict = load_graphs(graph_opath_generic, backoff_graph_opath_generic=None, entscr_key='weed', verbose=verbose, quiet=quiet)

    thiscate_all_concrete_eccs = set(concrete2vague_dict.keys())
    thiscate_all_vague_eccs = set(vague2concrete_dict.keys())
    thiscate_all_concrete_eccs_cnt = len(thiscate_all_concrete_eccs)
    thiscate_all_vague_eccs_cnt = len(thiscate_all_vague_eccs)
    print(f"Found {len(thiscate_all_concrete_eccs)} concrete ECCs and {len(thiscate_all_vague_eccs)} vague ECCs in the graphs; eval entries: {len(eval_entries)}")

    # Then evaluate the graphs

    # First evaluate the vague2concrete graph: for the matching of concrete ECCs (currently categories) there can be two metrics:
    # 1. Avarage likelihood of the best hit concrete ECCs from prediction;
    # 2. Mean Reciprocal Rank of the best hit concrete ECCs from prediction. (TODO: How to handle misses in mean reciprocal rank?)
    # The rationale is, although there can be multiple category forms, there is really just one category conceptually, so we just need to find the best match.
    if not quiet:
        print(f"Evaluating the graphs...")
    per_asin_eval_dicts = {x: [] for x in ENT_SCORES}
    empty_veccs_cnt = 0
    empty_ceccs_cnt = 0
    cecc_cobuy_identical_count = 0

    v2c_complete_miss_cnt = {x: 0 for x in ENT_SCORES}

    total_asins_cnt = len(eval_entries)
    asin_eval_start_time = time.time()

    predictions_ofp = gzip.open(predictions_opath, 'wt', encoding='utf8')
    total_round_trip_entry = 0
    total_c_property_count = {x: 0 for x in ENT_SCORES}
    hit_non_property_count = {x: 0 for x in ENT_SCORES}
    for aidx, curr_entry in enumerate(eval_entries):
        asin = curr_entry['asin']
        gold_veccs = curr_entry['intents']

        if aidx % 50 == 0:
            curr_time = time.time()
            durr = curr_time - asin_eval_start_time
            print(f'Evaluated {aidx} / {total_asins_cnt} ASINs; duration {durr // 60}min {durr % 60:.2f}sec;')
        
        gold_ceccs = curr_entry['ceccs']
        cobuy_ceccs = curr_entry['co_buy_ceccs']
        cobuy_ceccs = list(set(cobuy_ceccs).difference(set(gold_ceccs)))

        if len(cobuy_ceccs) == 0:
            cecc_cobuy_identical_count += 1
            cecc_cobuy_identical_flag = True
        else:
            cecc_cobuy_identical_flag = False

    
        if label_maxi_specificity is True:
            skip_gold_ceccs = set()
            for c in gold_ceccs:
                c_list = c.split(' ### ')
                c_parent = ' ### '.join(c_list[:-1])
                if c_parent in gold_ceccs:
                    skip_gold_ceccs.add(c_parent)
            gold_ceccs = [c for c in gold_ceccs if c not in skip_gold_ceccs]

            skip_cobuy_ceccs = set()
            for c in cobuy_ceccs:
                c_list = c.split(' ### ')
                c_parent = ' ### '.join(c_list[:-1])
                if c_parent in cobuy_ceccs:
                    skip_cobuy_ceccs.add(c_parent)
            cobuy_ceccs = [c for c in cobuy_ceccs if c not in skip_cobuy_ceccs]
        else:
            pass

        ctn_flag = False  # whether to skip this entry and continue to the next
        if len(gold_veccs) == 0:
            input_veccs = None
            empty_veccs_cnt += 1
            ctn_flag = True
        elif max_veccs_per_asin is not None and max_veccs_per_asin > 0 and len(gold_veccs) > max_veccs_per_asin:
            input_veccs = random.sample(gold_veccs, max_veccs_per_asin)
        else:
            input_veccs = gold_veccs
        if len(gold_ceccs) == 0:
            empty_ceccs_cnt += 1
            ctn_flag = True
        if ctn_flag:
            continue

        curr_entry['predicted_ceccs'] = {}
        curr_entry['twohop_predicted_ceccs'] = {}
        curr_entry['predicted_veccs'] = {}
        curr_entry['predicted_cobuy_ceccs'] = {}

        gold_veccs = {x[1]: x[2] for x in gold_veccs}
        input_veccs = {x[1]: x[2] for x in input_veccs} if input_veccs is not None else None

        for entscr_key in ENT_SCORES:
            predicted_ceccs = aggregate_correspondences(input_veccs, vague2concrete_dict, reduce=score_reduction, entscr_key=entscr_key, normalize=True, do_rank=True,
                                                        keep_size=keep_size, graph_target='c', skip_parent_ratio=skip_parent_ratio)
            if score_reduction == 'sum':
                two_hop_predicted_ceccs = aggregate_twohop_correspondences(input_veccs, vague2vague_dict, vague2concrete_dict, reduce=score_reduction, 
                                                                        entscr_key=entscr_key, normalize=True, do_rank=True, 
                                                                        max_medium_eccs=max_medium_eccs, keep_size=keep_size, twohop_epsilon=twohop_epsilon,
                                                                        graph_target='c', skip_parent_ratio=skip_parent_ratio)
            else:
                two_hop_predicted_ceccs = {}
            
            curr_entry['predicted_ceccs'][entscr_key] = predicted_ceccs if len(predicted_ceccs) > 0 else {k: x for k, (x, _) in cecc2ratio_dict.items()}
            curr_entry['twohop_predicted_ceccs'][entscr_key] = two_hop_predicted_ceccs

            enriched_predicted_ceccs = merge_two_predictions((predicted_ceccs, two_hop_predicted_ceccs), (1, twohop_weight))
            if len(enriched_predicted_ceccs) == 0:
                enriched_predicted_ceccs = {k: x for k, (x, _) in cecc2ratio_dict.items()}
            assert all([isinstance(enriched_predicted_ceccs[x], float) for x in enriched_predicted_ceccs])
            gold_cecc_distribution = {c: 1/len(gold_ceccs) for c in gold_ceccs}

            cecc_rmse, cecc_random_rmse = rmse(gold=gold_cecc_distribution, prediction=enriched_predicted_ceccs,
                            smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_concrete_eccs), do_random=True)
            
            cecc_majority_rmse, _ = rmse(gold=gold_cecc_distribution, prediction=cecc_maj_rmse_preds,
                                      smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_concrete_eccs), do_random=False)

            cecc_prediction_best2worst_rank = {p[0]: i+1 for (i, p) in enumerate(sorted(enriched_predicted_ceccs.items(), key=lambda x: x[1], reverse=True))}  # the best hit is ranked 0
            # any hit is a hit, so we just take the max of all categories.
            cecc_max_score = 0.0
            cecc_max_reciprocal_rank, cecc_mean_rr = 0, 0
            gold_match = ""
            cecc_majority_score = 0
            cecc_majority_rr = 0
            cecc_max_weighted_rr = 0
            posi_instances_sum = 0
            # TODO: This reciprocal rank weighting, should we use the training set stats or dev set stats? 
            # TODO: Evaluation sets themselves seem better, avoids reviewers' concerns on info leaking.
            v2c_hit_flag = False
            for g_cecc in gold_ceccs:
                if g_cecc in enriched_predicted_ceccs:
                    g_cecc_invcnt_factor = 1 / cecc2cnt_dict[g_cecc]
                    cecc_max_score = max(cecc_max_score, enriched_predicted_ceccs[g_cecc])
                    curr_g_rank = cecc_prediction_best2worst_rank[g_cecc]
                    cecc_mean_rr += 1.0/curr_g_rank
                    if 1.0/ curr_g_rank > cecc_max_reciprocal_rank:
                        gold_match = g_cecc
                    cecc_max_reciprocal_rank = max(cecc_max_reciprocal_rank, 1.0/curr_g_rank)
                    cecc_max_weighted_rr = max(cecc_max_weighted_rr, g_cecc_invcnt_factor/curr_g_rank)
                    posi_instances_sum += cecc2cnt_dict[g_cecc]
                    v2c_hit_flag = True
                if g_cecc in cecc2ratio_dict:
                    cecc_majority_rr = max(cecc_majority_rr, 1.0/(1+cecc2ratio_dict[g_cecc][1]))
                    cecc_majority_score = max(cecc_majority_score, cecc2ratio_dict[g_cecc][0])

            if not v2c_hit_flag:
                v2c_complete_miss_cnt[entscr_key] += 1
            else:
                if len(gold_match.split("#")) != 1:
                    total_c_property_count[entscr_key] += 1
                elif gold_match.split("#")[0] != "" and len(gold_match.split("#")) == 1:
                    hit_non_property_count[entscr_key] += 1

            cecc_random_rr = posi_instances_sum / total_cecc_instances_cnt
            cecc_random_wrr = len(gold_ceccs) / total_cecc_instances_cnt
            cecc_mean_rr /= len(gold_ceccs)

            # do the smoothing
            cecc_background_noise_scr = smooth_ratio / thiscate_all_concrete_eccs_cnt
            cecc_max_score *= (1 - smooth_ratio)
            cecc_max_score += cecc_background_noise_scr

            predicted_vecc_distribution = aggregate_correspondences(gold_ceccs, concrete2vague_dict, reduce=score_reduction, entscr_key=entscr_key, 
                                                                    normalize=True, do_rank=True, keep_size=keep_size, graph_target='v', skip_parent_ratio=skip_parent_ratio)
            curr_entry['predicted_veccs'][entscr_key] = predicted_vecc_distribution
            gold_vecc_distribution = {}
            for v in gold_veccs:
                gold_vecc_distribution[v] = gold_vecc_distribution.get(v, 0) + 1
            gold_vecc_sum = sum(gold_vecc_distribution.values())
            assert gold_vecc_sum > 0, f"Gold vague ECCs sum is 0 for ASIN {asin}!"
            for v in gold_vecc_distribution:
                gold_vecc_distribution[v] /= gold_vecc_sum
            vecc_kl_divergence, vecc_random_kl_divergence = kl_divergence(gold=gold_vecc_distribution, prediction=predicted_vecc_distribution, 
                                                smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_vague_eccs), do_random=True)
            vecc_rmse, vecc_random_rmse = rmse(gold=gold_vecc_distribution, prediction=predicted_vecc_distribution,
                            smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_vague_eccs), do_random=True)
            vecc_majority_rmse, _ = rmse(gold=gold_vecc_distribution, prediction=vecc_maj_rmse_preds,
                                        smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_vague_eccs), do_random=False)

            vecc_aggr_likelihood, vecc_random_aggr_likelihood, vecc_majority_aggr_likelihood = \
                aggr_likelihood(gold=gold_veccs, prediction=predicted_vecc_distribution, majorities=vecc_maj_rmse_preds,
                                                   smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_vague_eccs))

            vecc_rr, vecc_maj_rr = aggr_rr(gold=gold_veccs, prediction=predicted_vecc_distribution, majorities_ranks=vecc2ratio_dict,
                                             smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_vague_eccs))
            
            if do_round_trip_eval is True and not cecc_cobuy_identical_flag:
                total_round_trip_entry += 1
                round_trip_cecc_distribution = dict()
                rt_predicted_vecc_distribution = random_sized_sample(predicted_vecc_distribution, size=2000, portion=0.4)
                # avg_ceccs = []
                for v in rt_predicted_vecc_distribution:
                    # Here we shouldn't normalize for each individual vague ECC, because we want to aggregate the CECCs from all the vague ECCs.
                    curr_rt_ceccs = aggregate_correspondences([v], vague2concrete_dict, reduce=score_reduction, entscr_key=entscr_key, 
                                                              normalize=False, do_rank=True, keep_size=keep_size, graph_target='c', skip_parent_ratio=skip_parent_ratio)
                    for c in curr_rt_ceccs:
                        round_trip_cecc_distribution[c] = round_trip_cecc_distribution.get(c, 0) + curr_rt_ceccs[c] * predicted_vecc_distribution[v]
                # print(f"Total ceccs: {sum(avg_ceccs)}")
                # we remove the gold CECCs from the round-trip CECC distribution, so we only predict the new CECCs.
                for c in gold_ceccs:
                    assert c not in cobuy_ceccs, f"Gold CECC {c} is in cobuy CECCs for ASIN {asin}!"
                    if c in round_trip_cecc_distribution:
                        del round_trip_cecc_distribution[c]

                round_trip_cecc_sum = sum(round_trip_cecc_distribution.values())
                if round_trip_cecc_sum > 0:
                    for c in round_trip_cecc_distribution:
                        round_trip_cecc_distribution[c] /= round_trip_cecc_sum
                else:
                    assert all([x == 0 for x in round_trip_cecc_distribution.values()]), f"Round-trip CECC distribution sum is 0, but the distribution: {round_trip_cecc_distribution} is not empty for ASIN {asin}!"
                    round_trip_cecc_distribution = dict()
                
                tmp_dist = {k: v for k, v in sorted(round_trip_cecc_distribution.items(), key=lambda x: x[1], reverse=True)[:keep_size]}
                curr_entry['predicted_cobuy_ceccs'][entscr_key] = tmp_dist

                if len(cobuy_ceccs) == 0:
                    rt_cecc_max_score = None
                    rt_cecc_max_reciprocal_rank = None
                    rt_cecc_max_weighted_rr = None
                    rt_cecc_random_rr = None
                    rt_cecc_random_wrr = None
                    rt_cecc_rel_entr = None
                    rt_cecc_rmse = None
                    rt_cecc_random_rel_entr = None
                    rt_cecc_random_rmse = None
                else:
                    round_trip_cecc_best2worst_rank = {p[0]: i+1 for (i, p) in enumerate(sorted(round_trip_cecc_distribution.items(), key=lambda x: x[1], reverse=True))}  # the best hit is ranked 0
                    rt_cecc_max_score = 0.0
                    rt_cecc_max_reciprocal_rank = 0
                    rt_cecc_max_weighted_rr = 0
                    rt_posi_instances_sum = 0
                    for rt_gcecc in cobuy_ceccs:
                        if rt_gcecc in round_trip_cecc_distribution:
                            rt_gcecc_invcnt_factor = 1 / cecc2cnt_dict[rt_gcecc]
                            rt_cecc_max_score = max(rt_cecc_max_score, round_trip_cecc_distribution[rt_gcecc])
                            rt_curr_g_rank = round_trip_cecc_best2worst_rank[rt_gcecc]
                            rt_cecc_max_reciprocal_rank = max(rt_cecc_max_reciprocal_rank, 1.0/rt_curr_g_rank)
                            rt_cecc_max_weighted_rr = max(rt_cecc_max_weighted_rr, rt_gcecc_invcnt_factor/rt_curr_g_rank)
                            rt_posi_instances_sum += cecc2cnt_dict[rt_gcecc]
                    rt_cecc_random_rr = rt_posi_instances_sum / total_cecc_instances_cnt
                    rt_cecc_random_wrr = len(cobuy_ceccs) / total_cecc_instances_cnt

                    # do the smoothing
                    rt_cecc_background_noise_scr = cecc_background_noise_scr
                    rt_cecc_max_score *= (1 - smooth_ratio)
                    rt_cecc_max_score += rt_cecc_background_noise_scr

                    rt_cecc_gold_distribution = {x: 1.0/len(cobuy_ceccs) for x in cobuy_ceccs}

                    rt_cecc_rel_entr, rt_cecc_random_rel_entr = kl_divergence(gold=rt_cecc_gold_distribution, prediction=round_trip_cecc_distribution,
                                                        smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_concrete_eccs), do_random=True)

                    rt_cecc_rmse, rt_cecc_random_rmse = rmse(gold=rt_cecc_gold_distribution, prediction=round_trip_cecc_distribution,
                                                        smooth_ratio=smooth_ratio, keys_size=len(thiscate_all_concrete_eccs), do_random=True)
                    pass
            else:
                assert do_round_trip_eval is False or cecc_cobuy_identical_flag is True
                rt_cecc_max_score = None
                rt_cecc_max_reciprocal_rank = None
                rt_cecc_max_weighted_rr = None
                rt_cecc_random_rr = None
                rt_cecc_random_wrr = None
                rt_cecc_rel_entr = None
                rt_cecc_rmse = None
                rt_cecc_random_rel_entr = None
                rt_cecc_random_rmse = None
                curr_entry['predicted_cobuy_ceccs'][entscr_key] = None
            # TODO: when we see None for rt_cecc_XX, we should skip, should not count as 0.0!!!

            per_asin_eval_dicts[entscr_key].append({
                'asin': asin,
                'cecc_likelihood': cecc_max_score,
                'cecc_rr': cecc_max_reciprocal_rank,
                'cecc_mean_rr': cecc_mean_rr,
                'cecc_wrr': cecc_max_weighted_rr,
                'cecc_rmse': cecc_rmse,
                'cecc_random_rmse': cecc_random_rmse,
                'cecc_majority_rmse': cecc_majority_rmse,
                'cecc_majority_likelihood': cecc_majority_score,
                'cecc_majority_rr': cecc_majority_rr,
                'cecc_random_rr': cecc_random_rr,  # the probability that at least one gold cecc is randomly chosen from all concrete eccs, is the reciprocal rank. (geometric distributoin)
                'cecc_random_wrr': cecc_random_wrr,  # This random baseline is, if we sample a CECC according to its inverse count, what is the weighted reciprocal rank?
                'vecc_rel_entr': vecc_kl_divergence,
                'vecc_rmse': vecc_rmse,
                'vecc_random_rel_entr': vecc_random_kl_divergence,
                'vecc_random_rmse': vecc_random_rmse,
                'vecc_majority_rmse': vecc_majority_rmse,
                'vecc_al': vecc_aggr_likelihood,
                'vecc_random_al': vecc_random_aggr_likelihood,
                'vecc_majority_al': vecc_majority_aggr_likelihood,
                'vecc_rr': vecc_rr,
                'vecc_majority_rr': vecc_maj_rr,
                'rt_cecc_likelihood': rt_cecc_max_score,
                'rt_cecc_rr': rt_cecc_max_reciprocal_rank,
                'rt_cecc_wrr': rt_cecc_max_weighted_rr,
                'rt_cecc_random_rr': rt_cecc_random_rr,
                'rt_cecc_random_wrr': rt_cecc_random_wrr,
                'rt_cecc_rel_entr': rt_cecc_rel_entr,
                'rt_cecc_rmse': rt_cecc_rmse,
                'rt_cecc_random_rel_entr': rt_cecc_random_rel_entr,
                'rt_cecc_random_rmse': rt_cecc_random_rmse,
            })
            # print(f"entscr_key: {entscr_key}, rr: {cecc_max_reciprocal_rank}; cecc_prediction_best2worst_rank: {cecc_prediction_best2worst_rank.values()};")
        # print("")
        if not no_disk_cache:
            oline = json.dumps(curr_entry, ensure_ascii=False)
            predictions_ofp.write(oline + "\n")
        del curr_entry
    
    print(f"ASINs with empty vague ECCs: {empty_veccs_cnt}, empty concrete ECCs: {empty_ceccs_cnt};")
    print(f"ASINs with cecc complete missed by vague2concrete graph: {v2c_complete_miss_cnt};")
    print(f"The number of asin for round trip prediction is {total_round_trip_entry / len(ENT_SCORES)}")
    print(f"The number of hit cate has property is {total_c_property_count}")
    print(f"The number of hit cate don't have property is {hit_non_property_count}")
    print(f"Ratios of ASINS with cecc complete missed by vague2concrete graph:")
    for k, v in v2c_complete_miss_cnt.items():
        print(f"{k}: {v / len(eval_entries)*100:.2f}%;")
    predictions_ofp.close()

    out_dicts = {}
    
    for entscr_key in ENT_SCORES:
        print(f"Presenting evaluation results for {entscr_key}...")
        cecc_avg_likelihood = sum([d['cecc_likelihood'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_mrr = sum([d['cecc_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_mean_mrr = sum([d['cecc_mean_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_mean_wrr = sum([d['cecc_wrr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_avg_rmse = sum([d['cecc_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_random_avg_rmse = sum([d['cecc_random_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_majority_avg_rmse = sum([d['cecc_majority_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        
        cecc_random_mrr = sum([d['cecc_random_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_random_mean_wrr = sum(d['cecc_random_wrr'] for d in per_asin_eval_dicts[entscr_key]) / len(per_asin_eval_dicts[entscr_key])
        cecc_majority_avg_likelihood = sum([d['cecc_majority_likelihood'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        cecc_majority_mrr = sum([d['cecc_majority_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        print(f"Vague to Concrete mapping evaluation results: Average likelihood: {cecc_avg_likelihood} ({cecc_avg_likelihood*len(thiscate_all_concrete_eccs):.3f} times of random, {cecc_avg_likelihood/cecc_majority_avg_likelihood:.3f} times of majority baseline);")
        print(f"Max MRR: {round(cecc_mrr, 4)} ({(cecc_mrr+0.0000000001)/(cecc_random_mrr+0.0000000001):.3f} times of random, {(cecc_mrr+0.0000000001)/(cecc_majority_mrr+0.0000000001):.3f} times of majority baseline);")
        print(f"Mean MRR: {round(cecc_mean_mrr, 4)}")
        print(f"Avg RMSE: {cecc_avg_rmse}; random rmse: {cecc_random_avg_rmse}; majority rmse: {cecc_majority_avg_rmse};")
        print(f"Vague to Concrete mapping random baseline: Average likelihood: {1/len(thiscate_all_concrete_eccs)}, MRR: {cecc_random_mrr};")
        print(f"Vague2Concrete Mean Weighted Reciprocal Rank: {cecc_mean_wrr}; random: {cecc_random_mean_wrr}; Ratio: {(cecc_mean_wrr+0.0000000001)/(cecc_random_mean_wrr+0.0000000001):.3f};")
        print(f"Vague2Concrete Majority baseline: Average likelihood: {cecc_majority_avg_likelihood}, MRR: {cecc_majority_mrr};")

        effective_rel_entrs = [d['vecc_rel_entr'] for d in per_asin_eval_dicts[entscr_key] if d['vecc_rel_entr'] is not None]
        effective_rand_rel_entrs = [d['vecc_random_rel_entr'] for d in per_asin_eval_dicts[entscr_key] if d['vecc_random_rel_entr'] is not None]
        vecc_avg_rel_entropy = sum(effective_rel_entrs) / len(effective_rel_entrs)
        vecc_avg_rmse = sum([d['vecc_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_random_avg_rel_entropy = sum(effective_rand_rel_entrs) / len(effective_rand_rel_entrs)
        vecc_random_avg_rmse = sum([d['vecc_random_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_majority_avg_rmse = sum([d['vecc_majority_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_avg_al = sum([d['vecc_al'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_random_avg_al = sum([d['vecc_random_al'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_majority_avg_al = sum([d['vecc_majority_al'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_mrr = sum([d['vecc_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        vecc_majority_mrr = sum([d['vecc_majority_rr'] for d in per_asin_eval_dicts[entscr_key]]) / len(per_asin_eval_dicts[entscr_key])
        print(f"Concrete to Vague mapping evaluation result: Average relative entropy: {vecc_avg_rel_entropy}; Average RMSE: {vecc_avg_rmse};")
        print(f"Concrete to Vague mapping random baseline: Average relative entropy: {vecc_random_avg_rel_entropy}; Average RMSE: {vecc_random_avg_rmse};")
        print(f"Concrete to Vague mapping majority baseline: Average RMSE: {vecc_majority_avg_rmse};")
        print(f"Concrete2Vague Average Likelihood: {vecc_avg_al}; random: {vecc_random_avg_al}; majority: {vecc_majority_avg_al}; ratio to random: {vecc_avg_al/vecc_random_avg_al:.3f}; ratio to majority: {vecc_avg_al/vecc_majority_avg_al:.3f};")
        print(f"Concrete2Vague Mean Reciprocal Rank: {vecc_mrr}; majority: {vecc_majority_mrr}; ratio to majority: {vecc_mrr/vecc_majority_mrr:.3f};")

        rt_denominator = 0.01  # dummy value to avoid division by zero, subtracted below if no danger.
        for d in per_asin_eval_dicts[entscr_key]:
            if d['rt_cecc_likelihood'] is not None:
                assert d['rt_cecc_rr'] is not None
                assert d['rt_cecc_wrr'] is not None
                assert d['rt_cecc_random_rr'] is not None
                assert d['rt_cecc_random_wrr'] is not None
                assert d['rt_cecc_rmse'] is not None
                assert d['rt_cecc_random_rmse'] is not None
                if d['rt_cecc_rel_entr'] is None:
                    d['rt_cecc_rel_entr'] = 0
                if d['rt_cecc_random_rel_entr'] is None:
                    d['rt_cecc_random_rel_entr'] = 0
                rt_denominator += 1
            else:
                assert d['rt_cecc_rr'] is None
                assert d['rt_cecc_wrr'] is None
                assert d['rt_cecc_random_rr'] is None
                assert d['rt_cecc_random_wrr'] is None
                assert d['rt_cecc_rel_entr'] is None
                assert d['rt_cecc_rmse'] is None
                assert d['rt_cecc_random_rel_entr'] is None
                assert d['rt_cecc_random_rmse'] is None
                # These zeros also do not participate in the denominator, therefore they are harmless.
                d['rt_cecc_likelihood'] = 0
                d['rt_cecc_rr'] = 0
                d['rt_cecc_wrr'] = 0
                d['rt_cecc_random_rr'] = 0
                d['rt_cecc_random_wrr'] = 0
                d['rt_cecc_rel_entr'] = 0
                d['rt_cecc_rmse'] = 0
                d['rt_cecc_random_rel_entr'] = 0
                d['rt_cecc_random_rmse'] = 0
        if rt_denominator > 0.01:
            rt_denominator -= 0.01

        rt_cecc_avg_likelihood = sum([d['rt_cecc_likelihood'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_mrr = sum([d['rt_cecc_rr'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_mean_wrr = sum([d['rt_cecc_wrr'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_random_mrr = sum([d['rt_cecc_random_rr'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_random_mean_wrr = sum(d['rt_cecc_random_wrr'] for d in per_asin_eval_dicts[entscr_key]) / rt_denominator
        rt_cecc_avg_rel_entropy = sum([d['rt_cecc_rel_entr'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_avg_rmse = sum([d['rt_cecc_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_random_avg_rel_entropy = sum([d['rt_cecc_random_rel_entr'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        rt_cecc_random_avg_rmse = sum([d['rt_cecc_random_rmse'] for d in per_asin_eval_dicts[entscr_key]]) / rt_denominator
        print(f"Round-Trip Concrete2Concrete Recommendation evaluation results: Average likelihood: {rt_cecc_avg_likelihood} ({rt_cecc_avg_likelihood*len(thiscate_all_concrete_eccs):.3f} times of random), MRR: {rt_cecc_mrr} ({(rt_cecc_mrr+0.0000000001)/(rt_cecc_random_mrr+0.0000000001):.3f} times of random, {(rt_cecc_mrr+0.0000000001)/(cecc_majority_mrr+0.0000000001):.3f} times of majority baseline);")
        print(f"Round-Trip Concrete2Concrete Recommendation random baseline: Average likelihood: {1/len(thiscate_all_concrete_eccs)}, MRR: {rt_cecc_random_mrr};")
        print(f"Round-Trip Concrete2Concrete Mean Weighted Reciprocal Rank: {rt_cecc_mean_wrr}; random: {rt_cecc_random_mean_wrr}; Ratio: {(rt_cecc_mean_wrr+0.0000000001)/(rt_cecc_random_mean_wrr+0.0000000001):.3f};")
        print(f"Round-Trip Concrete2Concrete Recommendation evaluation result: Average relative entropy: {rt_cecc_avg_rel_entropy}; Average RMSE: {rt_cecc_avg_rmse};")
        print(f"Round-Trip Concrete2Concrete Recommendation random baseline: Average relative entropy: {rt_cecc_random_avg_rel_entropy}; Average RMSE: {rt_cecc_random_avg_rmse};")

        curr_dict = {
            'cecc_avg_likelihood': cecc_avg_likelihood,
            'cecc_mrr': cecc_mrr,
            'cecc_random_mrr': cecc_random_mrr,
            'vecc_avg_rel_entropy': vecc_avg_rel_entropy,
            'vecc_avg_rmse': vecc_avg_rmse,
            'vecc_random_avg_rel_entropy': vecc_random_avg_rel_entropy,
            'vecc_random_avg_rmse': vecc_random_avg_rmse,
            'rt_cecc_avg_likelihood': rt_cecc_avg_likelihood,
            'rt_cecc_mrr': rt_cecc_mrr,
            'rt_cecc_random_mrr': rt_cecc_random_mrr,
            'rt_cecc_avg_rel_entropy': rt_cecc_avg_rel_entropy,
            'rt_cecc_avg_rmse': rt_cecc_avg_rmse,
            'rt_cecc_random_avg_rel_entropy': rt_cecc_random_avg_rel_entropy,
            'rt_cecc_random_avg_rmse': rt_cecc_random_avg_rmse,
            'per_asin_eval_dicts': per_asin_eval_dicts[entscr_key],
        }
        out_dicts[entscr_key] = curr_dict

    print(f"Saving the evaluation results to {scores_opath}...")
    with open(scores_opath, 'w', encoding='utf8') as ofp:
        json.dump(out_dicts, ofp, indent=2, ensure_ascii=False)
    
    if not quiet:
        print(f"Done!")
    return


def batch_calc_rep_from_model(terms: list, term_cache: list, term_mapping, model, tokenizer, device, batch_size: int):
    padding_tok_id = tokenizer.pad_token_id
    for batch_start_id in range(0, len(terms), batch_size):
        # if batch_start_id // batch_size < 28600:  # TODO: debug
        #     continue
        if batch_start_id // batch_size % 100 == 0 and batch_start_id // batch_size > 0:
            print(f"Processing batch {batch_start_id // batch_size} / {len(terms) // batch_size + 1}...")
        curr_batch = terms[batch_start_id:batch_start_id+batch_size]
        inputs = tokenizer(curr_batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.last_hidden_state
            # print(f"last hidden state size: {last_hidden_states.shape}")
            for i, term in enumerate(curr_batch):
                mask = inputs.input_ids[i]!=padding_tok_id
                curr_len = torch.sum(mask, dim=0).int()
                curr_rep = torch.sum(last_hidden_states[i][1:curr_len-1], dim=0) / (curr_len-2)  # remove the [CLS] and [SEP] tokens
                term_cache.append(curr_rep)
                if term_mapping is not None:
                    term_mapping[term] = len(term_cache) - 1
    return


def get_rep_from_model(term: str, model, tokenizer, device):
    # Below are the steps to calculate the representation of a term, in case the term is not in the cache
    inputs = tokenizer(term, return_tensors='pt')
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state[0, 1:-1, :]  # remove the [CLS] and [SEP] tokens
        curr_rep = torch.mean(last_hidden_states, dim=0)
    return curr_rep


def gpt_rank_v2c(veccs: list, prd_ceccs: list, model_name: str, temperature: float, top_p: float,
                 sleep_after_query: float):  # for GPT, we don't know the lobprobs of the sequences, so we can't weight the scores with the original entailment scores, it has to be one or the other.

    veccs_set = list(set([x[1] for x in veccs]))
    veccs_set = ["\""+v+"\"" for v in veccs_set]
    veccs_str = ', '.join(random.sample(veccs_set, min(100000, len(veccs_set))))

    prd_ceccs, prd_cecc_scrs = zip(*prd_ceccs)

    prd_cecc_str = "\n".join([f"{id+1}. {cecc}" for id, cecc in enumerate(prd_ceccs)])

    template = f"A product is suitable for the following purposes: \n{veccs_str}\n\n" \
    f"Please rank the following categories in order of likelihood that the product belongs to them (most likely to least likely): \n" \
    f"{prd_cecc_str}\n\n" \
    f"Answer:\n" \
    f"1."


    # template = (f"A product is suitable for the following purposes: \n{veccs_str}\n\n" \
    # f"Please rank the following possible categories based on how likely the product belongs to (most likely to least likely): \n" \
    # f"{prd_cecc_str}\n\n" \
    # f"Answer:\n" \
    # f"1."
    # )
    # f"Please rank the following possible categories based on how likely the product belongs to (most likely to least likely): \n" \
    # f"Answer with the re-rank index\n" \
    
    if model_name.startswith("gpt-3.5"):
        prompt_dict = wrap_prompt_chat(template, model_name, 256, temperature, top_p)
    else:
        prompt_dict = wrap_prompt_completion(template, model_name, 256, temperature, top_p)
    response = None
    for _try in range(MAX_NUM_RETRIES):
        try:
            if model_name.startswith("gpt-3.5"):
                response = openai.ChatCompletion.create(**prompt_dict)
            else:
                response = openai.Completion.create(**prompt_dict)
            time.sleep(sleep_after_query)
            break
        except Exception as e:
            print(f"Error: {e}")
            if _try == MAX_NUM_RETRIES-1:
                pass
            else:
                time.sleep(args.sleep_after_query)
                print(f"Retrying...")
                continue

    if response is None:
        print(f"Error: response is None", file=sys.stderr)
        return [], 0, 0
    else:
        if model_name.startswith("gpt-3.5"):
            ret_text = response['choices'][0]['message']['content'] # type: ignore
            num_toks = response['usage']['total_tokens']  # type: ignore
        else:
            ret_text = response['choices'][0]['text'].strip(' ') # type: ignore
            num_toks = 0
        # print(f"Returned text: {ret_text}")

    ret_list = process_list_from_output(ret_text)
    # print(f"Processed vals: {ret_list}")

    overlaps = set(ret_list).intersection(set(prd_ceccs))
    print(f"Overlaps: {len(overlaps)}; ret_list: {len(ret_list)}; prd_ceccs: {len(prd_ceccs)}")

    return ret_list, num_toks, len(overlaps)  # , response


def gpt_rank_rt(orig_ceccs: list, prd_veccs: list, rt_prd_ceccs: list, model_name: str, temperature: float, 
                top_p: float, sleep_after_query: float):  # for GPT, we don't know the lobprobs of the sequences, so we can't weight the scores with the original entailment scores, it has to be one or the other.

    prd_veccs_set = list(set(prd_veccs))
    prd_veccs_set = ["\""+v+"\"" for v in prd_veccs_set]
    prd_veccs_str = ', '.join(prd_veccs_set)

    orig_ceccs_set = list(set(orig_ceccs))
    orig_ceccs_set = ["\""+v+"\"" for v in orig_ceccs_set]
    orig_ceccs_str = ', '.join(orig_ceccs_set)


    rt_prd_ceccs, rt_prd_cecc_scrs = zip(*rt_prd_ceccs)

    rt_prd_cecc_str = "\n".join([f"{id}. {cecc}" for id, cecc in enumerate(rt_prd_ceccs)])

    template = f"A product can be described as the followings: {orig_ceccs_str}\n" \
    f"It is found to be good for the following purposes: \n{prd_veccs_str}\n\n" \
    f"Customers who buy this product may also buy which other products? Re-rank the following candidates from most likely to least likely: \n" \
    f"{rt_prd_cecc_str}\n\n" \
    f"Answer:\n" \
    f"1."
    
    if model_name.startswith("gpt-3.5"):
        prompt_dict = wrap_prompt_chat(template, model_name, 256, temperature, top_p)
    else:
        prompt_dict = wrap_prompt_completion(template, model_name, 256, temperature, top_p)
    response = None
    for _try in range(MAX_NUM_RETRIES):
        try:
            if model_name.startswith("gpt-3.5"):
                response = openai.ChatCompletion.create(**prompt_dict)
            else:
                response = openai.Completion.create(**prompt_dict)
            time.sleep(args.sleep_after_query)
            break
        except Exception as e:
            print(f"Error: {e}")
            if _try == MAX_NUM_RETRIES-1:
                pass
            else:
                time.sleep(sleep_after_query)
                print(f"Retrying...")
                continue

    if response is None:
        print(f"Error: response is None", file=sys.stderr)
        return [], 0
    else:
        if model_name.startswith("gpt-3.5"):
            ret_text = response['choices'][0]['message']['content'] # type: ignore
            num_toks = response['usage']['total_tokens']  # type: ignore
        else:
            ret_text = response['choices'][0]['text'].strip(' ') # type: ignore
            num_toks = 0
        # print(f"Returned text: {ret_text}")

    ret_list = process_list_from_output(ret_text)
    # print(f"Processed vals: {ret_list}")

    return ret_list, num_toks  # , response


def bertlike_rank(gold_eccs, prd_eccs, model, tokenizer, batch_size, device, score_reduction: str, rerank_alpha: float):  
    gold_eccs_set = list(set(gold_eccs))
    # print(f"G: {gold_eccs_set}")

    prd_eccs, prd_ecc_scrs = zip(*prd_eccs)

    prd_ecc_scrs = torch.tensor(prd_ecc_scrs, dtype=torch.float32, device=device)

    gold_ecc_reps = []
    prd_ecc_reps = []

    batch_calc_rep_from_model(gold_eccs_set, gold_ecc_reps, None, model, tokenizer, device, batch_size=batch_size)
    batch_calc_rep_from_model(list(prd_eccs), prd_ecc_reps, None, model, tokenizer, device, batch_size=batch_size)


    gold_ecc_reps = torch.stack(gold_ecc_reps, dim=0)  # shape: (num_ceccs, rep_dim)
    prd_ecc_reps = torch.stack(prd_ecc_reps, dim=0)  # shape: (num_ceccs, rep_dim)
    # print(f"X: {gold_ecc_reps}")
    # print(f"Y: {prd_ecc_reps}")
    similarity_scores = torch.matmul(gold_ecc_reps, prd_ecc_reps.t())  # shape: (veccs, prd_ceccs)
    # print(f"Z: {similarity_scores}")
    if score_reduction == 'sum':
        similarity_scores = torch.sum(similarity_scores, dim=0, keepdim=False)
    elif score_reduction == 'max':
        similarity_scores = torch.max(similarity_scores, dim=0, keepdim=False)
    elif score_reduction == 'mean':
        similarity_scores = torch.mean(similarity_scores, dim=0, keepdim=False)
    else:
        raise AssertionError
    # normalize the similarity_scores
    # print(f'A: {similarity_scores}')
    similarity_scores = similarity_scores / torch.sum(similarity_scores)
    # print(f'B: {similarity_scores}')
    bertscr_dct = dict(zip(prd_eccs, similarity_scores.tolist()))
    # Geometrically average the similarity scores with the entailment scores
    similarity_scores = torch.log(similarity_scores) * rerank_alpha + torch.log(prd_ecc_scrs) * (1-rerank_alpha)
    # print(f'C: {similarity_scores}')

    similarity_ranking = torch.argsort(similarity_scores, dim=0, descending=True).tolist()

    reranked_prd_ceccs = [prd_eccs[i] for i in similarity_ranking]

    # print(f"alpha: {rerank_alpha}")
    # print(f"reranked preds: {reranked_prd_ceccs}")
    # print(f"original preds: {prd_eccs}")
    # print(f"reranked scores: {similarity_scores.tolist()}")
    # print(f"original scores: {prd_ecc_scrs}")

    return reranked_prd_ceccs, bertscr_dct


def lm_reranking_main(predictions_out_path, entscr_key, rerank_n: int, model_type: str, model_name: str, device: str,
                       batch_size: int, score_reduction: str, rerank_alpha: float, do_round_trip: bool,
                       sleep_after_query: float, do_hparam: bool):

    if model_type == 'bertlike':
        model = AutoModel.from_pretrained(f'/home/teddy/lms/{model_name}')
        tokenizer = AutoTokenizer.from_pretrained(f'/home/teddy/lms/{model_name}', max_seq_length=500)
        assert model is not None
        model.to(device)
        model.eval()
        print(f"Model {model_name} loaded.")
    elif model_type == 'gpt':
        model = None
        tokenizer = None
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    # predictions_dicts = load_gz_jsonl(predictions_out_path)
            
    prd_cecc_recranks, prd_rank = [], {"h":[], "d": [], "n": []}
    prd_vecc_recranks = []
    prd_rt_cecc_recranks = []

    prev_prd_cecc_recranks, prev_rank = [], {"h":[], "d": [], "n": []}
    prev_prd_vecc_recranks = []
    prev_prd_rt_cecc_recranks = []

    rerank_helpful_cnt = 0
    rerank_detrimental_cnt = 0
    rerank_neutral_cnt = 0
    rt_rerank_helpful_cnt = 0
    rt_rerank_detrimental_cnt = 0
    rt_rerank_neutral_cnt = 0
    not_rerank = 0
    lidx = 0
    total_num_toks = 0
    total_reranked_generations = 0.00000000000000001
    total_reranked_overlaps = 0.00000000000000001
    # print(f"predictions_dicts[0]: {predictions_dicts[0]}")
    total_entry = []
    with gzip.open(predictions_out_path, 'rt', encoding='utf-8') as rfp:
        for lidx, line in enumerate(rfp):
            
            if lidx and lidx % 100 == 0:
                print(f"The lidx is {lidx}")
                print(f"Loading gz jsonl line {lidx}")
                # break
                # if model_type == 'gpt':
                #     print(f"Current total number of tokens used: {total_num_toks}")
                #     print(f"Current statistics on hit in reranked generations: {total_reranked_overlaps} / {total_reranked_generations} = {total_reranked_overlaps*100 / total_reranked_generations:.2f}%")
            if len(line) < 2:
                continue
            try:
                entry = json.loads(line)
                total_entry.append(entry)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Line {lidx}: {line}")
                raise e
    random.seed(42)
    # entry_sample = random.sample(total_entry, 100)
    # print(f"Delete the previous prediction set {len(total_entry)}")
    # del total_entry
    
    with gzip.open(predictions_out_path + '.reranked.gz', 'wt', encoding='utf-8') as ofp:
        for entry in total_entry:
            gold_veccs = entry['intents']
            gold_ceccs = [x.lower() for x in entry['ceccs']]
            cobuy_ceccs = entry['co_buy_ceccs']
            predicted_ceccs = entry['predicted_ceccs'][entscr_key]
            predicted_veccs = entry['predicted_veccs'][entscr_key]
        
            prd_ceccs_all = [(k.lower(), v) for (k, v) in sorted(predicted_ceccs.items(), key=lambda item: item[1], reverse=True)]
            prd_ceccs_top = prd_ceccs_all[:rerank_n]
            prd_veccs_all = [(k.lower(), v) for (k, v) in sorted(predicted_veccs.items(), key=lambda item: item[1], reverse=True)]
            prd_veccs_top = prd_veccs_all[:rerank_n]
            if do_round_trip and entry['predicted_cobuy_ceccs'][entscr_key] is not None:
                predicted_cobuy_ceccs = entry['predicted_cobuy_ceccs'][entscr_key]
                prd_cobuy_ceccs_all = [(k.lower(), v) for (k, v) in sorted(predicted_cobuy_ceccs.items(), key=lambda item: item[1], reverse=True)]
                prd_cobuy_ceccs_top = prd_cobuy_ceccs_all[:rerank_n]
            else:
                prd_cobuy_ceccs_all = []
                prd_cobuy_ceccs_top = []
            # print("For all gold ceccs..............")
            prev_recrank, prev_likelihood = calc_recrank(predictions=[k for (k, v) in prd_ceccs_all],
                                                        prd_scrs={k: v for (k, v) in prd_ceccs_all}, 
                                                        references=gold_ceccs)
            # print("For top gold ceccs..............")
            prev_top_recrank, prev_top_likelihood = calc_recrank(predictions=[k for (k, v) in prd_ceccs_top],
                                                        prd_scrs={k: v for (k, v) in prd_ceccs_top}, 
                                                        references=gold_ceccs)
            
            prev_prd_cecc_recranks.append(prev_recrank)
            
            if len(prd_ceccs_top) == 0:
                rerank_neutral_cnt += 1
                entry[f'prd_ceccs_top_{model_type}_reranked'] = []
                if model_type == 'gpt':
                    pass
                elif model_type == 'bertlike':
                    raise NotImplementedError
                    entry['bertlike_scrs'] = {}
                else:
                    raise ValueError(f"Invalid model type: {model_type}")
                prd_cecc_recranks.append(0.0)
                assert prev_recrank == 0.0
            elif prev_top_recrank == 0:
                print("Not in top 10")
                print(gold_ceccs)
                not_rerank += 1
                entry[f'prd_ceccs_top_{model_type}_reranked'] = [k for (k, v) in prd_ceccs_top]
                prd_cecc_recranks.append(prev_recrank)
                continue
            else:
                # Now we do the reranking for V->C evaluation
                if model_type == 'gpt':
                    print("here")
                    reranked_cands, curr_num_toks, len_overlap = gpt_rank_v2c(veccs=gold_veccs, prd_ceccs=prd_ceccs_top, model_name=model_name, temperature=0.0, 
                                                top_p=1.0, sleep_after_query=sleep_after_query)  #, rerank_alpha=rerank_alpha)
                    print(f"curr num tokens: {curr_num_toks}")
                    total_num_toks += curr_num_toks
                    total_reranked_generations += len(reranked_cands)
                    total_reranked_overlaps += len_overlap
                elif model_type == 'bertlike':
                    # raise NotImplementedError
                    assert model is not None and tokenizer is not None
                    reranked_cands, bertlike_scrs = bertlike_rank(gold_eccs=gold_veccs, prd_eccs=prd_ceccs_top, model=model, tokenizer=tokenizer, batch_size=batch_size,
                                    device=device, score_reduction=score_reduction, rerank_alpha=rerank_alpha)
                    entry['bertlike_scrs'] = bertlike_scrs
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                entry[f'prd_ceccs_top_{model_type}_reranked'] = reranked_cands
                new_prd_ceccs_all = [k for (k, v) in prd_ceccs_all]
                new_prd_ceccs_all = reranked_cands + new_prd_ceccs_all[rerank_n:]
                print("For the ceccs after re-ranking...........")
                prd_cecc_recrank, _ = calc_recrank(predictions=new_prd_ceccs_all, prd_scrs=None,
                                                    references=gold_ceccs)
                # print(prd_cecc_recrank)
                # print(prev_recrank)

                prd_cecc_recranks.append(prd_cecc_recrank)

                if prd_cecc_recrank > prev_recrank:
                    rerank_helpful_cnt += 1
                    print(f"helpful! prvious is {1/prev_recrank if prev_recrank != 0 else -1} , now is {1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1}")
                    prd_rank['h'].append(1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1)
                    prev_rank['h'].append(1/prev_recrank if prev_recrank != 0 else -1)
                    
                elif prd_cecc_recrank < prev_recrank:
                    rerank_detrimental_cnt += 1
                    print(f"detrimental!previous is {1/prev_recrank if prev_recrank != 0 else -1}, now is {1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1}")
                    prd_rank['d'].append(1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1)
                    prev_rank['d'].append(1/prev_recrank if prev_recrank != 0 else -1)
                else:
                    rerank_neutral_cnt += 1
                    print(f"neutral! previous rank is {1/prev_recrank if prev_recrank != 0 else -1}, now is {1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1}")
                    prd_rank['n'].append(1/prd_cecc_recrank if prd_cecc_recrank != 0 else -1)
                    prev_rank['n'].append(1/prev_recrank if prev_recrank != 0 else -1)

            # Now we do the reranking for C->V evaluation (not implemented)
            pass

            if do_round_trip and len(prd_cobuy_ceccs_top) > 0:
                prev_rt_recrank, prev_rt_likelihood = calc_recrank(predictions=[k for (k, v) in prd_cobuy_ceccs_all],
                                                                prd_scrs={k: v for (k, v) in prd_cobuy_ceccs_all},
                                                                references=cobuy_ceccs)
                prev_prd_rt_cecc_recranks.append(prev_rt_recrank)

                # Now we do the reranking for round-trip evaluation
                prd_veccs_top_keys = [k for (k, v) in prd_veccs_top]
                if model_type == 'gpt':
                    rt_reranked_cands, curr_num_toks = gpt_rank_rt(orig_ceccs=gold_ceccs, prd_veccs=prd_veccs_top_keys, rt_prd_ceccs=prd_cobuy_ceccs_top, 
                                                    model_name=model_name, temperature=0.0, top_p=1.0, sleep_after_query=sleep_after_query)  #, rerank_alpha=rerank_alpha)
                    print(f"Round trip current num tokens: {curr_num_toks}")
                    total_num_toks += curr_num_toks
                elif model_type == 'bertlike':
                    rt_reranked_cands, rt_bertlike_scrs = bertlike_rank(gold_eccs=gold_ceccs, prd_eccs=prd_cobuy_ceccs_top, model=model, tokenizer=tokenizer, 
                                                        batch_size=batch_size, device=device, score_reduction=score_reduction, rerank_alpha=rerank_alpha)
                    entry['rt_bertlike_scrs'] = rt_bertlike_scrs
                else:
                    raise ValueError(f"Invalid model type: {model_type}")

                entry[f'prd_rt_ceccs_top_{model_type}_reranked'] = rt_reranked_cands
                new_prd_rt_ceccs_all = [k for (k, v) in prd_cobuy_ceccs_all]
                new_prd_rt_ceccs_all = rt_reranked_cands + new_prd_rt_ceccs_all[rerank_n:]

                prd_rt_cecc_recrank, prd_rt_cecc_likelihood = calc_recrank(predictions=new_prd_rt_ceccs_all, 
                                                                        prd_scrs=None,
                                                                        references=cobuy_ceccs)
                prd_rt_cecc_recranks.append(prd_rt_cecc_recrank)
                if prd_rt_cecc_recrank > prev_rt_recrank:
                    rt_rerank_helpful_cnt += 1
                elif prd_rt_cecc_recrank < prev_rt_recrank:
                    rt_rerank_detrimental_cnt += 1
                else:
                    rt_rerank_neutral_cnt += 1
            elif do_round_trip:
                entry[f'prd_rt_ceccs_top_{model_type}_reranked'] = []
                if model_type == 'gpt':
                    pass
                elif model_type == 'bertlike':
                    entry['rt_bertlike_scrs'] = {}
                else:
                    raise ValueError(f"Invalid model type: {model_type}")
            else:
                pass
            ofp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if model_type == 'gpt':
        print(f"Current total number of tokens used: {total_num_toks}")
        print(f"Current statistics on hit in reranked generations: {total_reranked_overlaps} / {total_reranked_generations} = {total_reranked_overlaps*100 / total_reranked_generations:.2f}%")

    print(f"Evaluation results re-ranking: {predictions_out_path}:")
    print(f"Rerank alpha: {rerank_alpha}")

    print(len(prd_cecc_recranks))
    print(len(prev_prd_cecc_recranks))
    v2c_mrr = np.mean(prd_cecc_recranks)
    
    print(f"v2c_mrr: {v2c_mrr}")
    print("The rank distribution after llm_reranking")
    res_lit = []
    for i, (k, v) in enumerate(prd_rank.items()):
        print(f"For the {k}: the distribution is {dict(Counter(v))}")
        res_lit.extend(v)
    print(f"Total distribution is {dict(Counter(res_lit))}")

    if do_round_trip:
        if len(prd_rt_cecc_recranks) > 0:
            rt_c2c_mrr = np.mean(prd_rt_cecc_recranks)
        else:
            rt_c2c_mrr = 0
        print(f"rt_c2c_mrr: {rt_c2c_mrr}")
    else:
        pass

    prev_v2c_mrr = np.mean(prev_prd_cecc_recranks)
    print(f"prev_v2c_mrr: {prev_v2c_mrr}")
    print("Previous rank distribution before llm rerank")
    res_lit = []
    for i, (k,v) in enumerate(prev_rank.items()):
        print(f"For the {k}: the distribution is {dict(Counter(v))}")
        res_lit.extend(v)
    print(f"Total distribution is {dict(Counter(res_lit))}")

    # prev_rt_c2c_mrr = np.mean(prev_prd_rt_cecc_recranks)
    # print(f"prev_rt_c2c_mrr: {prev_rt_c2c_mrr}")

    print(f"Total number of entries: {lidx}")

    print(f"rerank helpful cnt: {rerank_helpful_cnt}")
    print(f"rerank detrimental cnt: {rerank_detrimental_cnt}")
    print(f"rerank neutral cnt: {rerank_neutral_cnt}")
    print(f"Round-Trip rerank helpful cnt: {rt_rerank_helpful_cnt}")
    print(f"Round-Trip rerank detrimental cnt: {rt_rerank_detrimental_cnt}")
    print(f"Round-Trip rerank neutral cnt: {rt_rerank_neutral_cnt}")

    if 'Clothing' in predictions_out_path:
        num_thres = 50
    else:
        num_thres = 100

    if model_type == 'bertlike' and do_hparam:
        best_mrr = 0.0
        best_alpha = None
        for try_alpha_id in range(num_thres+1):
            try_alpha = try_alpha_id / num_thres
            recranks = []
            with gzip.open(predictions_out_path + '.reranked.gz', 'rt', encoding='utf-8') as ifp:
                for line in ifp:
                    if len(line) < 2:
                        continue
                    entry = json.loads(line)
                    bertlike_scrs_top = entry['bertlike_scrs']
                    gold_ceccs = entry['cecc']
                    predicted_ceccs = entry['predicted_ceccs'][entscr_key]
                    prd_ceccs_all = [(k, v) for (k, v) in sorted(predicted_ceccs.items(), key=lambda item: item[1], reverse=True)]
                    prd_ceccs_top = {k: v for (k, v) in prd_ceccs_all[:rerank_n]}
                    prd_ceccs_topkeys = list(prd_ceccs_top.keys())
                    assert set(bertlike_scrs_top.keys()) == set(prd_ceccs_topkeys)
                    bertlike_scr_list = torch.tensor([bertlike_scrs_top[k] for k in prd_ceccs_topkeys])
                    prd_scrs_list = torch.tensor([prd_ceccs_top[k] for k in prd_ceccs_topkeys])
                    merged_scrs_list = torch.log(bertlike_scr_list)*try_alpha + torch.log(prd_scrs_list)*(1-try_alpha)
                    similarity_ranking = torch.argsort(merged_scrs_list, dim=0, descending=True).tolist()
                    reranked_prd_ceccs = [prd_ceccs_topkeys[i] for i in similarity_ranking]
                    prd_cecc_allkeys = [k for (k, v) in prd_ceccs_all]
                    reranked_prd_ceccs_all = reranked_prd_ceccs + prd_cecc_allkeys[rerank_n:]
                    prd_cecc_recrank, _ = calc_recrank(predictions=reranked_prd_ceccs_all,
                                                        prd_scrs=None,
                                                        references=gold_ceccs)
                    recranks.append(prd_cecc_recrank)
            mrr = np.mean(recranks)
            print(f"alpha: {try_alpha}, mrr: {mrr}")
            if mrr > best_mrr:
                best_mrr = mrr
                best_alpha = try_alpha
        print(f"best_alpha: {best_alpha}, best_mrr: {best_mrr}")
    else:
        best_alpha = None
    
    # overwrite the predictions file with the one with reranked predictions
    # dump_gz_jsonl(predictions_dicts, predictions_out_path)
    # shutil.move(predictions_out_path + '.reranked', predictions_out_path)
    return best_alpha


def weight_direct_twohops(predictions_out_path, entscr_key, v2cobuy_flag: bool):
    predictions_dicts_dev = load_gz_jsonl(predictions_out_path % 'dev')
    print(f"Total number of dev entries: {len(predictions_dicts_dev)}")
    print(f"entscr_key: {entscr_key}")
    print(f"Analyzing prediction file: {predictions_out_path}...")

    best_mrr = 0.0
    mll_at_best_mrr = 0.0
    best_twohop_weight = 0.0

    for w_idx in range(26):
        twohop_weight = w_idx / 100
        recranks = []
        likelihoods = []

        for eidx, entry in enumerate(predictions_dicts_dev):
            # if eidx % 1000 == 0:
            #     print(f"Processing entry {eidx}...")
            # print(list(entry.keys()))
            # print(entry.keys())
            gold_ceccs = entry['co_buy_ceccs'] if v2cobuy_flag else entry['ceccs']
            prd_ceccs_onehop = entry['predicted_ceccs'][entscr_key]
            prd_ceccs_twohops = entry['twohop_predicted_ceccs'][entscr_key]
            prd_ceccs_dct = merge_two_predictions((prd_ceccs_onehop, prd_ceccs_twohops), (1, twohop_weight))
            prd_ceccs_lst = [k for (k, v) in sorted(prd_ceccs_dct.items(), key=lambda x: x[1], reverse=True)[:100]]
            prd_cecc_recrank, prd_cecc_likelihood = calc_recrank(predictions=prd_ceccs_lst, prd_scrs=prd_ceccs_dct,
                                                                 references=gold_ceccs)
            recranks.append(prd_cecc_recrank)
            likelihoods.append(prd_cecc_likelihood)
        print(f"twohop_weight: {twohop_weight}")
        mrr = np.mean(recranks)
        mll = np.mean(likelihoods)
        print(f"MRR: {mrr}")
        print(f"Avg Likelihood: {mll}")
        if mrr > best_mrr:
            best_mrr = mrr
            mll_at_best_mrr = mll
            best_twohop_weight = twohop_weight
    
    print(f'Best twohop_weight: {best_twohop_weight}')
    print(f'Best Dev MRR: {best_mrr}')
    print(f'Dev Avg Likelihood at best MRR: {mll_at_best_mrr}')
    print("")
    del predictions_dicts_dev

    predictions_dicts_test = load_gz_jsonl(predictions_out_path % 'test')
    print(f"Total number of test entries: {len(predictions_dicts_test)}")

    test_recranks = []
    test_likelihoods = []
    for test_twohop_weight in [0.0, best_twohop_weight]:
        for eidx, entry in enumerate(predictions_dicts_test):
            if eidx % 1000 == 0:
                print(f"Processing entry {eidx}...")
            # print(entry.keys())
            gold_ceccs = entry['co_buy_ceccs'] if v2cobuy_flag else entry['ceccs']
            prd_ceccs_onehop = entry['predicted_ceccs'][entscr_key]
            prd_ceccs_twohops = entry['twohop_predicted_ceccs'][entscr_key]
            prd_ceccs_dct = merge_two_predictions((prd_ceccs_onehop, prd_ceccs_twohops), (1, test_twohop_weight))
            prd_ceccs_lst = [k for (k, v) in sorted(prd_ceccs_dct.items(), key=lambda x: x[1], reverse=True)[:100]]
            prd_cecc_recrank, prd_cecc_likelihood = calc_recrank(predictions=prd_ceccs_lst, prd_scrs=prd_ceccs_dct,
                                                                    references=gold_ceccs)
            test_recranks.append(prd_cecc_recrank)
            test_likelihoods.append(prd_cecc_likelihood)
        print(f"Test twohop_weight: {test_twohop_weight}")
        print(f"Test MRR: {np.mean(test_recranks)}")
        print(f"Test Avg Likelihood: {np.mean(test_likelihoods)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_root', type=str, default='%s_graphs/OWM%s/%s/')
    parser.add_argument('--graph_out_fn', type=str, default='%s.OWM%s.json')

    parser.add_argument('--cate_name', type=str, default='Home_and_Kitchen', help='Name of the broad category, e.g. Home_and_Kitchen')
    parser.add_argument('--score_version', type=str, default='local', help='Version of the ECC score, e.g. local, global-weeds-4')
    parser.add_argument('--occ_weight_mode', type=str, default='count', 
                        help='Measure of occurrence strength for vague-related graphs, namely the D function in Weed et al; could be [count, comb, pmi, p_ecc or geometric].')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    parser.add_argument('--data_root', type=str, default='./folkscope')
    parser.add_argument('--meta_root', type=str, default=os.path.join(os.path.expanduser('~'), 'amazon_metas/'))
    parser.add_argument('--assembled_root', type=str, default='./folkscope/assembled_evaldata/')
    parser.add_argument('--source_fn', type=str, default='%s_intents_%s.json')
    parser.add_argument('--meta_cate_fn', type=str, default='meta_%s_wpnames_cleaned.json')
    parser.add_argument('--cecc2cnt_fn', type=str, default='meta_%s_cecc2cnt_train.json')
    parser.add_argument('--vecc2cnt_fn', type=str, default='meta_%s_intent2cnt_train.json')
    parser.add_argument('--assembled_fn', type=str, default='assembled_%s_%s.json')
    parser.add_argument('--cecc_set_fn', type=str, default='meta_%s_ceccset.json')
    parser.add_argument('--vecc_set_fn', type=str, default='meta_%s_intentset.json')
    parser.add_argument('--view2buy_rel_weight', type=float, default=0, help='Relative weight for co-view compared to co-buy in the ECC graph.')
    parser.add_argument('--max_medium_eccs', type=int, default=10, help='Maximum number of medium ECCs to be considered for each vague ECC when doing two-hop.')
    parser.add_argument('--keep_size', type=int, default=300, help="Maximum number of predicted target ECCs to keep for each entry.")
    parser.add_argument('--twohop_weight', type=float, default=0.0, help="Weight for two-hop ECC connections relative to one hop weights.")
    parser.add_argument('--twohop_epsilon', type=float, default=0.01, help="Epsilon for two-hop ECC connections.")
    parser.add_argument('--skip_parent_ratio', type=float, default=None, help='the minimum ratio of child cecc\'s score relative to the parent\'s, in order for the parent to be considered redundant.')
    parser.add_argument('--label_maxi_specificity', action='store_true', help='Whether to test with the most specific labels possible for each entry.')
    
    parser.add_argument('--v2cobuy_flag', action='store_true', help='Whether to use cobuy ECCs as gold standard for vague ECC connections in benchmarking.')

    parser.add_argument('--score_reduction', type=str, default='sum', choices=['max', 'avg', 'sum'], help='Reduction method for predicted categories from multiple vague-ecc instances.')
    parser.add_argument('--smooth_ratio', type=float, default=0.01, help='Smooth ratio for unrecorded targets.')
    parser.add_argument('--max_veccs_per_asin', type=int, default=None, help='Maximum number of vague ECCs to be considered for each entry, to control amount of context for v2v efficacy test.')

    parser.add_argument('--output_fn', type=str, default='eval_%s_OWM%s_%s%.1f_MAXIN%s.json')
    parser.add_argument('--eval_subset', type=str, default='dev', choices=['dev', 'test', 'all'], help='Subset to evaluate on: [dev, test]')
    parser.add_argument('--task', type=str, default='auto', help='[auto, human_pre, human_post]')

    parser.add_argument('--do_round_trip', action='store_true', help='Whether to do round-trip evaluation.')
    parser.add_argument('--cecc_equally_true', action='store_true', 
                        help='This flag has no use here, it\'s a placeholder for flag alignment in BASH file.')
    parser.add_argument('--backoff_to_local', action='store_true', help='Whether to backoff to local ECC score when global ECC score is not available.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    parser.add_argument('--model_name', type=str, default='bert-large-uncased', help='Name of the pretrained model for LM baseline.')
    parser.add_argument('--lm_device', type=str, default='cuda:0', help='Device for LM baseline.')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--rerank_model_type', type=str, default='bertlike', choices=['bertlike', 'gpt'])
    parser.add_argument('--rerank_n', type=int, default=10, help='Number of candidates to rerank.')
    parser.add_argument('--rerank_alpha', type=float, default=0.5, help='Alpha for reranking.')
    parser.add_argument('--rerank_scrkey', type=str, default='weed', help='Key for reranking.')

    parser.add_argument('--sleep_after_query', type=float, default=2.0, help='Sleep time after each query (only for GPT).')

    parser.add_argument('--no_disk_cache', action='store_true', help='Whether to disable disk cache.')
    parser.add_argument('--entropy', action='store_true', help='Whether to load the entropy grah')

    args = parser.parse_args()
    if args.verbose:
        print(args)
    assert args.smooth_ratio > 0, f"Smooth ratio should be positive! Otherwise ZeroDivisionError could occur."

    if 'local' in args.score_version:
        ENT_SCORES = ['cos', 'weed', 'lin', 'binc']
    elif 'global' in args.score_version:
        ENT_SCORES = ['weed']
    else:
        raise ValueError(f"Unknown score version: {args.score_version}")
    
    if args.skip_parent_ratio is None:
        if args.label_maxi_specificity:
            # best setup from dev sets of Office_Product and Patio_Lawn_and_Garden
            args.skip_parent_ratio = 0.25
        else:
            args.skip_parent_ratio = 1.0

    if args.task == 'assemble':
        for subset in ['dev', 'test']:
            print(f"Assembling the evaluation data for {subset}...")
            source_path = os.path.join(args.data_root, args.source_fn % (args.cate_name, subset))

            meta_cate_name = args.cate_name.rstrip('_05_05').rstrip('_05').rstrip('_09_09').rstrip('_09').rstrip('_07')
            meta_cate_path = os.path.join(args.meta_root, args.meta_cate_fn%meta_cate_name)
            assembled_out_path = os.path.join(args.data_root, args.assembled_fn % (args.cate_name, subset))
            assemble_eval_data(source_path=source_path, c_path=meta_cate_path, assembled_opath=assembled_out_path, verbose=args.verbose, quiet=args.quiet)
            
    elif args.task == 'auto':
        # if args.entropy:
        args.graph_root = os.path.join(args.data_root, "entropy", args.graph_root)
        # else:
        #     args.graph_root = os.path.join(args.data_root, args.graph_root)
        graph_root = args.graph_root % (args.cate_name, args.occ_weight_mode, args.score_version)
        graph_out_fn = args.graph_out_fn % (f'%s_{args.score_version}', args.occ_weight_mode)  # the remaining %s is for the file type
        graph_out_path_generic = os.path.join(graph_root, graph_out_fn)
        print(f"Automatically evaluating the ECC graphs for {args.cate_name} from {graph_out_path_generic}...")
        source_path = os.path.join(args.data_root, args.source_fn % (args.cate_name, args.eval_subset))
        
        cecc2cnt_fpath = os.path.join(args.data_root, args.cecc2cnt_fn % args.cate_name)  # We use the cecc cnt stats from the training set, since it is only used to calculate random baseline.
        vecc2cnt_fpath = os.path.join(args.data_root, args.vecc2cnt_fn % args.cate_name)
        print(f"Skip parent ratio: {args.skip_parent_ratio}")
        
        max_veccs_str = f"{args.max_veccs_per_asin}" if args.max_veccs_per_asin is not None else "null"
        scores_out_path = os.path.join(graph_root, args.output_fn % (args.eval_subset, args.occ_weight_mode, args.score_reduction, args.smooth_ratio, args.max_veccs_per_asin))
        assert scores_out_path.endswith('.json')
        predictions_out_path = scores_out_path.replace('.json', '_predictions.json.gz')
        assert predictions_out_path != scores_out_path
        assembled_out_path = os.path.join(args.data_root, args.assembled_fn % (args.cate_name.split("0")[0].strip("_"), args.eval_subset))
        
        auto_eval_main(assembled_opath=assembled_out_path, graph_opath_generic=graph_out_path_generic,
                       cecc2cnt_path=cecc2cnt_fpath, vecc2cnt_path=vecc2cnt_fpath, score_reduction=args.score_reduction,
                        smooth_ratio=args.smooth_ratio, max_medium_eccs=args.max_medium_eccs, scores_opath=scores_out_path, predictions_opath=predictions_out_path,
                        do_round_trip_eval=args.do_round_trip, debug=args.debug, verbose=args.verbose, quiet=args.quiet, keep_size=args.keep_size, 
                        twohop_weight=args.twohop_weight, twohop_epsilon=args.twohop_epsilon, max_veccs_per_asin=args.max_veccs_per_asin,
                        skip_parent_ratio=args.skip_parent_ratio, label_maxi_specificity=args.label_maxi_specificity, no_disk_cache=args.no_disk_cache)
    elif args.task == 'lm_rerank':
        if args.entropy:
            graph_root = os.path.join(args.data_root, "entropy", args.graph_root) % (args.cate_name, args.occ_weight_mode, args.score_version)
        else:
            graph_root = os.path.join(args.data_root, args.graph_root) % (args.cate_name, args.occ_weight_mode, args.score_version)
        scores_out_path = os.path.join(graph_root, args.output_fn % ('%s', args.occ_weight_mode, args.score_reduction, args.smooth_ratio, 'None'))
        predictions_out_path = scores_out_path.replace('.json', '_predictions.json.gz')
        assert predictions_out_path != scores_out_path

        if args.score_reduction != 'sum':
            print(f"Warning! The score reduction method is not sum, but {args.score_reduction}. This may cause unexpected behavior in the reranking process.", file=sys.stderr)

        if args.eval_subset in ['dev', 'test']:
            do_hparam = args.eval_subset == 'dev'
            scores_out_path = scores_out_path % args.eval_subset
            predictions_out_path = predictions_out_path % args.eval_subset
            lm_reranking_main(predictions_out_path, entscr_key=args.rerank_scrkey, rerank_n=args.rerank_n, model_type=args.rerank_model_type, 
                           model_name=args.model_name, device=args.lm_device, batch_size=args.batch_size, score_reduction=args.score_reduction,
                           rerank_alpha=args.rerank_alpha, do_round_trip=args.do_round_trip, sleep_after_query=args.sleep_after_query,
                           do_hparam=do_hparam)
        else:
            assert args.eval_subset == 'all'
            scores_out_path_dev = scores_out_path % 'dev'
            predictions_out_path_dev = predictions_out_path % 'dev'
            best_alpha = lm_reranking_main(predictions_out_path_dev, entscr_key=args.rerank_scrkey, rerank_n=args.rerank_n, model_type=args.rerank_model_type,
                            model_name=args.model_name, device=args.lm_device, batch_size=args.batch_size, score_reduction=args.score_reduction,
                            rerank_alpha=args.rerank_alpha, do_round_trip=args.do_round_trip, sleep_after_query=args.sleep_after_query,
                            do_hparam=True)
            assert best_alpha is not None
            print(f"Best Alpha for {args.cate_name} dev set: {best_alpha}!")
            print(f"Running on test set...")
            scores_out_path_test = scores_out_path % 'test'
            predictions_out_path_test = predictions_out_path % 'test'
            lm_reranking_main(predictions_out_path_test, entscr_key=args.rerank_scrkey, rerank_n=args.rerank_n, model_type=args.rerank_model_type,
                            model_name=args.model_name, device=args.lm_device, batch_size=args.batch_size, score_reduction=args.score_reduction,
                            rerank_alpha=best_alpha, do_round_trip=args.do_round_trip, sleep_after_query=args.sleep_after_query,
                            do_hparam=False)
    elif args.task == 'twohop_hparam':
        graph_root = os.path.join(args.data_root, args.graph_root) % (args.cate_name, args.occ_weight_mode, args.score_version)
        
        max_veccs_str = f"{args.max_veccs_per_asin}" if args.max_veccs_per_asin is not None else "null"
        scores_out_path = os.path.join(graph_root, args.output_fn % ('%s', args.occ_weight_mode, args.score_reduction, args.smooth_ratio, args.max_veccs_per_asin))
        predictions_out_path = scores_out_path.replace('.json', '_predictions.json.gz')
        weight_direct_twohops(predictions_out_path, args.rerank_scrkey, v2cobuy_flag=args.v2cobuy_flag)

    elif args.task == 'human_pre':
        raise NotImplementedError
    elif args.task == 'human_post':
        raise NotImplementedError
    else:
        raise ValueError('Unknown task: %s' % args.task)



    