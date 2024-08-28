import argparse 
import os 
import pandas as pd
import shutil
parser = argparse.ArgumentParser() 

parser.add_argument(
    "--img_dir",
    type=str,
    required=True,
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--ext",
    type=str,
    default='png'
)

if __name__ == '__main__':
    args = parser.parse_args()
    idxs = list()
    prompts = list()
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    for idx, f in enumerate(os.listdir(args.img_dir)):
        prompt = f.split('-')[-1].replace(f".{args.ext}", "")
        prompt = prompt.split('|')[0]
        shutil.copy(os.path.join(args.img_dir, f), os.path.join(args.out_dir, "images", f"{idx}.{args.ext}"))
        idxs.append(f"{idx}.{args.ext}")
        prompts.append(prompt)

    df = pd.DataFrame(dict(filename=idxs, prompts=prompts))
    df.to_csv(os.path.join(args.out_dir, 'result.csv'))
    


