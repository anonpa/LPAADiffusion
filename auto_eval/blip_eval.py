import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from rich.progress import track
sys.path.append(".")
sys.path.append("..")
import os
from .imagenet_utils import get_embedding_for_prompt, imagenet_templates
import pandas as pd
import stanza
from nltk.tree import Tree
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
import random 
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError



def get_token_alignment_map(tree, tokens):
    if tokens is None:
        return {i:[i] for i in range(len(tree.leaves())+1)}
        
    def get_token(token):
        return token[:-4] if token.endswith("</w>") else token

    idx_map = {}
    j = 0
    max_offset = np.abs(len(tokens) - len(tree.leaves()))
    mytree_prev_leaf = ""
    for i, w in enumerate(tree.leaves()):
        token = get_token(tokens[j])
        idx_map[i] = [j]
        if token == mytree_prev_leaf+w:
            mytree_prev_leaf = ""
            j += 1
        else:
            if len(token) < len(w):
                prev = ""
                while prev + token != w:
                    prev += token
                    j += 1
                    token = get_token(tokens[j])
                    idx_map[i].append(j)
                    # assert j - i <= max_offset
            else:
                mytree_prev_leaf += w
                j -= 1
            j += 1
    idx_map[i+1] = [j]
    return idx_map


def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        if tree.label() == 'NP' and n_leaves > 1:
            sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
        return sub_nps
    
    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            if i == j: continue
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    if lowest_only:
        all_nps = lowest_nps

    if len(all_nps) == 0:
        all_nps = []
        spans = []
    else:
        all_nps, spans = map(list, zip(*all_nps))
    if full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(min(idx_map[start]), min(idx_map[end]))] + spans

    return all_nps, spans, lowest_nps


@dataclass
class EvalConfig:
    # output_path: Path = Path("~/Projects/decompose/Guided_Syn/output/")
    output_path = None
    metrics_save_path: Path = Path("./metrics/")
    input_dir: Path = Path("/home/zhlu6105/Projects/decompose/Guided_Syn/output") 
    csv_path: Path = Path("/home/zhlu6105/Projects/decompose/Guided_Syn/result.csv")
    eval_partial: bool = True
    truncate: bool = True
    ext: str = 'png'

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)





# @pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    exp_name = str(config.input_dir).split('/')[-1]
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")

    print("Loading BLIP model...")
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_image_text_matching", model_type="pretrain", is_eval=True, device=device)
    blip_model = blip_model.float()
    print("Done.")

    # prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    files = os.listdir(config.input_dir)
    print(f"Running on {len(files)} prompts...")


    results_per_prompt = {}
    itcs = list() 
    min_itcs = list() 
    avg_itcs = list() 
    itms = list()
    min_itms = list() 
    avg_itms = list() 


    df = pd.read_csv(config.csv_path)

    for img in track(files):
        # prompt = img.replace(f'.{config.ext}', '').split('-')[-1].strip()
        img_path = os.path.join(config.input_dir, img)
        img_path = str(img_path)
        # prompt = df.loc[df['filename']==img]['prompt'].values[0]
        try:
            prompt = df.loc[df['filename']==img]['prompt'].values[0]
        except:
            prompt = df.loc[df['filename']==img]['prompts'].values[0]
        prompts = [prompt]
        if config.eval_partial:
            try:
                doc = nlp(prompt)
                mytree = Tree.fromstring(str(doc.sentences[0].constituency))
                # tokens = model.cond_stage_model.tokenizer.tokenize(prompts[0])
                tokens = None
                prompt_parts, spans, noun_chunk = get_all_nps(mytree, prompt, tokens)
                prompts = prompt_parts # remove the first sentence as it is the full sentence
            except:
                print(f"failed to parse: {prompt}")
                continue





        # get all images for the given prompt
        image_paths = [img_path]
        images = [Image.open(p) for p in image_paths]
        with torch.no_grad():
            # extract prompt embeddings
            # prompt_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)

            blip_input_images = [vis_processors["eval"](image).unsqueeze(0).to(device,) for image in images]


            single_itcs = list() 
            single_itms = list() 

            for text in prompts:
                blip_input_texts = txt_processors["eval"](text)
                itm_output = blip_model({"image": blip_input_images[0], "text_input": blip_input_texts}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1).cpu()
                itc_score = blip_model({"image": blip_input_images[0], "text_input": blip_input_texts}, match_head="itc").cpu()

                single_itms.append(itm_scores[:, 1].numpy())
                single_itcs.append(itc_score.numpy())

            itms.append(single_itms[0])
            itcs.append(single_itcs[0])

            if len(prompts) > 1:
                min_itcs.append(np.min(single_itcs[1:]))
                min_itms.append(np.min(single_itms[1:]))
                avg_itcs.append(np.asarray(single_itcs[1:]).mean())
                avg_itms.append(np.asarray(single_itms[1:]).mean())
            else:
                min_itcs.append(np.min(itms[-1]))
                min_itms.append(np.min(itms[-1]))
                avg_itcs.append(np.mean(itcs[-1]))
                avg_itms.append(np.mean(itcs[-1]))

    aggregated_results = {
        'itcs': np.asarray(itcs).mean(),
        'itcs_min': np.asarray(min_itcs).mean(),
        'itcs_avg': np.asarray(avg_itcs).mean(),
        'itms': np.asarray(itms).mean(),
        'itms_min': np.asarray(min_itms).mean(),
        'itms_avg': np.asarray(avg_itms).mean(),
    }

    return aggregated_results

    # with open(config.metrics_save_path / f"blip_raw_metrics_{exp_name}.json", 'w') as f:
    #     json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    # with open(config.metrics_save_path / f"blip_aggregated_metrics_{exp_name}.json", 'w') as f:
    #     json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_text_similarities(result_dict):
    all_averages = [result_dict[prompt]['text_similarities'] for prompt in result_dict]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std


# if __name__ == '__main__':
#     run()
