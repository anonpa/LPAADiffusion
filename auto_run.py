import argparse
from run import generate, save_image, load_model
import os
from auto_eval import run_clip_eval, CLIPConfig, run_blip_eval, BLIPConfig
import re
import pandas as pd
import shutil
from rich.progress import track
import time
import torch

def objective(configs):
    csv_path = configs['csv_path']
    output_directory = configs['output_directory']
    model_name = configs['model_name']
    dataset_path = configs['dataset_path']
    seed = configs['seed']

    os.makedirs(output_directory, exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)


    pipe = load_model(model_name, device='cuda:0')

    f_names = list()
    save_prompts = list()
    with open(dataset_path, 'r') as f:
        prompts = f.read().splitlines()


    # prompts = random.sample(prompts, 10)
    all_mem = list()
    start = time.time()
    for idx, single_prompt in enumerate(track(prompts)):
        single_prompt = re.sub('\s+',' ',single_prompt).replace('.', '')
        image = generate(pipe, single_prompt, seed, None, 256, False, device='cuda:0', lpaa_config=configs, probar=True)
        mem = torch.cuda.mem_get_info()#[1]/(1024**2)
        used_mem = mem[1] - mem[0]
        used_mem = used_mem / (1024**2)
        all_mem.append(used_mem)
        file_name = save_image(image, str(idx), seed, output_directory, show_date=False)
        save_prompts.append(single_prompt)
        f_names.append(file_name) 

    print(f'time used: {time.time()-start}')
    data = dict(filename=f_names, prompt=save_prompts)
    df = pd.DataFrame(data)
    df.to_csv(csv_path)
        



    # evaluate 
    score = 0
    clip_eval_config = CLIPConfig(input_dir=output_directory, csv_path=csv_path)
    blip_eval_config = BLIPConfig(input_dir=output_directory, csv_path=csv_path)

    print(f"dataset {dataset_path} \n csv saved to {csv_path}\n images saved to {output_directory}")
    clip_result = run_clip_eval(clip_eval_config)
    blip_result = run_blip_eval(blip_eval_config)
    print(clip_result)
    print(blip_result)
    print(f"mem used: {sum(all_mem)/len(all_mem)}")

    for _, v in clip_result.items():
        score += v

    for _, v in blip_result.items():
        score += v
    return {'score': score}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=float,
        default=100.,
    )
    parser.add_argument(
        "--preserve_scale",
        type=float,
        default=10.,
    )
    parser.add_argument(
        "--lse_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--align_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--area_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--disalign_scale",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--single_pass",
        action="store_true"
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=1,
        help='the caching interval'

    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--target_res",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--target_layers",
        nargs="+",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='CompVis/stable-diffusion-v1-5'
    )
    args = parser.parse_args()

    model_name = args.model_path 
    seed = 42

    
    output_directory = os.path.join(args.output_dir, args.exp_name, 'images')
    os.makedirs(output_directory, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.exp_name, 'result.csv')
    lpaa_config = dict(
                    scale=args.scale,
                    preserve_scale =args.preserve_scale,
                    lse_scale = args.lse_scale,
                    align_scale = args.align_scale,
                    area_scale = args.area_scale,
                    disalign_scale = args.disalign_scale,
                    single_pass = args.single_pass,
                    cache=args.cache,
                    beta = args.beta,
                    csv_path = csv_path, 
                    output_directory = output_directory, 
                    model_name = model_name, 
                    dataset_path = args.prompt_file,
                    seed = seed,
                    target_res = args.target_res,
                    target_layers = args.target_layers,
                )

    objective(lpaa_config)

