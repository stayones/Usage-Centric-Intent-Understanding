import json
import gzip
import copy
from typing import Optional, Union


def load_gz_jsonl(path):
    predictions_dicts = []
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for lidx, line in enumerate(f):
            if lidx % 1000 == 0:
                print(f"Loading gz jsonl line {lidx}")
            if len(line) < 2:
                continue
            try:
                item = json.loads(line)
                for k in item['predicted_ceccs']['weed']:
                    assert isinstance(item['predicted_ceccs']['weed'][k], float)
                predictions_dicts.append(json.loads(line))
            except Exception as e:
                print(f"Error: {e}")
                print(f"Line {lidx}: {line}")
                raise e
    
    return predictions_dicts


def dump_gz_jsonl(predictions_dicts, path):
    with gzip.open(path, 'wt', encoding='utf8') as fp:
        for lidx, pred_dict in enumerate(predictions_dicts):
            if lidx % 1000 == 0:
                print(f"Dumping gz jsonl line {lidx}")
            oline = json.dumps(pred_dict, ensure_ascii=False) + '\n'
            fp.write(oline)  # type: ignore


def calc_recrank(predictions: list, prd_scrs: Union[dict, None], references: list):
    max_reciprocal_rank = 0
    max_likelihood = 0
    assert prd_scrs is None or all([p in prd_scrs for p in predictions])
    for ref in references:
        if ref in predictions:
            curr_rr = 1 / (predictions.index(ref) + 1)
            if curr_rr > max_reciprocal_rank:
                max_reciprocal_rank = curr_rr
            if prd_scrs is not None:
                if prd_scrs[ref] > max_likelihood:
                    max_likelihood = prd_scrs[ref]
    return max_reciprocal_rank, max_likelihood
            

def load_graphs(graph_opath_generic: str, backoff_graph_opath_generic: Optional[str], entscr_key: str, verbose: bool, quiet: bool):
    # then load the graph (here only the concrete2vague and vague2concrete graphs are evaluated, since they are the end products; the other two graphs are for the enrichment of these two)
    # UPDATE: now also include the vague2vague graph, which is used for explicit two-hop reasoning.
    concrete2vague_dict = {}
    vague2concrete_dict = {}
    vague2vague_dict = {}
    if not quiet:
        print(f"Loading concrete2vague graph from {graph_opath_generic % 'concrete2vague'}")
    with open(graph_opath_generic % 'concrete2vague', 'r', encoding='utf8') as gfp:
        for lidx, line in enumerate(gfp):
            if lidx % 100000 == 0 and verbose:
                print('Loaded %d concrete2vague mappings' % lidx)
            jobj = json.loads(line)
            concrete2vague_dict[jobj['concrete']] = jobj['oedges']
    
    if not quiet:
        print(f"Loading vague2concrete graph from {graph_opath_generic % 'vague2concrete'}")
    with open(graph_opath_generic % 'vague2concrete', 'r', encoding='utf8') as gfp:
        for lidx, line in enumerate(gfp):
            if lidx % 100000 == 0 and verbose:
                print('Loaded %d vague2concrete mappings' % lidx)
            jobj = json.loads(line)
            vague2concrete_dict[jobj['vague']] = jobj['oedges']
    
    if not quiet:
        print(f"Loading vague2vague graph from {graph_opath_generic % 'vague2vague'}")
    with open(graph_opath_generic % 'vague2vague', 'r', encoding='utf8') as gfp:
        for lidx, line in enumerate(gfp):
            if lidx % 100000 == 0 and verbose:
                print('Loaded %d vague2vague mappings' % lidx)
            jobj = json.loads(line)
            vague2vague_dict[jobj['vague']] = jobj['oedges']
    
    if backoff_graph_opath_generic is not None:
        if not quiet:
            print(f"Backoff concrete2vague graph from {backoff_graph_opath_generic % 'concrete2vague'}")
        with open(backoff_graph_opath_generic % 'concrete2vague', 'r', encoding='utf8') as gfp:
            for lidx, line in enumerate(gfp):
                if lidx % 100000 == 0 and verbose:
                    print('Loaded %d concrete2vague mappings' % lidx)
                jobj = json.loads(line)
                if jobj['concrete'] not in concrete2vague_dict:
                    concrete2vague_dict[jobj['concrete']] = jobj['oedges']
                else:
                    for vecc in jobj['oedges']:
                        if vecc not in concrete2vague_dict[jobj['concrete']] or (entscr_key is not None and concrete2vague_dict[jobj['concrete']][vecc][entscr_key] == 0):
                            concrete2vague_dict[jobj['concrete']][vecc] = jobj['oedges'][vecc]
                        else:
                            pass
        
        if not quiet:
            print(f"Backoff vague2concrete graph from {backoff_graph_opath_generic % 'vague2concrete'}")
        with open(backoff_graph_opath_generic % 'vague2concrete', 'r', encoding='utf8') as gfp:
            for lidx, line in enumerate(gfp):
                if lidx % 100000 == 0 and verbose:
                    print('Loaded %d vague2concrete mappings' % lidx)
                jobj = json.loads(line)
                if jobj['vague'] not in vague2concrete_dict:
                    vague2concrete_dict[jobj['vague']] = jobj['oedges']
                else:
                    for cecc in jobj['oedges']:
                        if cecc not in vague2concrete_dict[jobj['vague']] or vague2concrete_dict[jobj['vague']][cecc][entscr_key] == 0:
                            vague2concrete_dict[jobj['vague']][cecc] = jobj['oedges'][cecc]
                        else:
                            pass
        
        if not quiet:
            print(f"Backoff vague2vague graph from {backoff_graph_opath_generic % 'vague2vague'}")
        with open(backoff_graph_opath_generic % 'vague2vague', 'r', encoding='utf8') as gfp:
            for lidx, line in enumerate(gfp):
                if lidx % 100000 == 0 and verbose:
                    print('Loaded %d vague2vague mappings' % lidx)
                jobj = json.loads(line)
                if jobj['vague'] not in vague2vague_dict:
                    vague2vague_dict[jobj['vague']] = jobj['oedges']
                else:
                    for vecc in jobj['oedges']:
                        if vecc not in vague2vague_dict[jobj['vague']] or vague2vague_dict[jobj['vague']][vecc][entscr_key] == 0:
                            vague2vague_dict[jobj['vague']][vecc] = jobj['oedges'][vecc]
                        else:
                            pass
    else:
        pass

    if not quiet:
        print(f"Graphs loaded!")
    
    return concrete2vague_dict, vague2concrete_dict, vague2vague_dict

def kl_divergence(gold: Dict[str, float], prediction: Dict[str, float], smooth_ratio: float, keys_size: int, do_random: bool):
    """
    Calculate the KL divergence between two distributions. Here it is okay for the gold and predicted distributions to be not normalized,
    as the scipy.stats.entropy function will normalize them.
    gold: a dictionary mapping ECCs to their probabilities
    prediction: a dictionary mapping ECCs to their probabilities
    """
    gold_keys = set(gold.keys())
    pred_keys = set(prediction.keys())
    common_keys = gold_keys.intersection(pred_keys)
    goldonly_keys = gold_keys.difference(common_keys)
    predonly_keys = pred_keys.difference(common_keys)
    gold_list = [gold[k] for k in common_keys]
    pred_list = [prediction[k]*(1-smooth_ratio) for k in common_keys]

    background_noise_scr = smooth_ratio / keys_size
    for k in goldonly_keys:
        gold_list.append(gold[k])
        pred_list.append(background_noise_scr)
    for k in predonly_keys:
        gold_list.append(0.0)
        pred_list.append(prediction[k])
    pred_list = [x if x > background_noise_scr else background_noise_scr for x in pred_list]
    assert len(gold_list) == len(pred_list)
    assert len(gold_keys) > 0
    assert sum(gold_list) > 0.0 and sum(pred_list) > 0.0, f"gold_list: {gold_list}, pred_list: {pred_list}, gold: {gold}, prediction: {prediction}"
    res = scipy.stats.entropy(gold_list, pred_list)

    if do_random:
        r_gold_list = []
        for k in gold:
            r_gold_list.append(gold[k])
        while len(r_gold_list) < keys_size:
            r_gold_list.append(0.0)
        rand_list = [1.0/keys_size for _ in range(keys_size)]
        rand_res = scipy.stats.entropy(r_gold_list, rand_list)
    else:
        rand_res = None
    
    if math.isinf(res):
        res = None
        rand_res = None

    return res, rand_res

def rmse(gold: Dict[str, float], prediction: Dict[str, float], smooth_ratio: float, keys_size: int, do_random: bool):
    """
    Calculate the RMSE between two distributions. Here it is okay for the gold and predicted distributions to be not normalized,
    as the scipy.stats.entropy function will normalize them.
    gold: a dictionary mapping ECCs to their probabilities
    prediction: a dictionary mapping ECCs to their probabilities

    In FolkScope, their RMSE measure considers only the items that are in the gold set, and ignores the items that are in the prediction set but not in the gold set.
    Link: https://github.com/HKUST-KnowComp/FolkScope/blob/9aa37e951687051fe5b28153394e49506821e61f/src/recommendation/run_WnD_co_buy.py#L17
    """
    gold_keys = set(gold.keys())
    pred_keys = set(prediction.keys())
    common_keys = gold_keys.intersection(pred_keys)
    goldonly_keys = gold_keys.difference(common_keys)
    predonly_keys = pred_keys.difference(common_keys)
    gold_list = [gold[k] for k in common_keys]
    pred_list = [prediction[k]*(1-smooth_ratio) for k in common_keys]
    background_noise_scr = smooth_ratio / keys_size
    for k in goldonly_keys:
        gold_list.append(gold[k])
        pred_list.append(background_noise_scr)
    # for k in predonly_keys:
    #     gold_list.append(0.0)
    #     pred_list.append(prediction[k])
    assert len(gold_list) == len(pred_list)
    res = mean_squared_error(gold_list, pred_list, squared=False)

    if do_random:
        r_gold_list = []
        for k in gold:
            r_gold_list.append(gold[k])
        # while len(r_gold_list) < keys_size:
        #     r_gold_list.append(0.0)
        # rand_list = [1.0/keys_size for _ in range(keys_size)]
        rand_list = [1.0/keys_size for _ in range(len(r_gold_list))]
        assert len(r_gold_list) == len(rand_list) == len(gold_list)
        rand_res = mean_squared_error(r_gold_list, rand_list, squared=False)
    else:
        rand_res = None

    return res, rand_res