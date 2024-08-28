import os 
import shutil
import argparse 
from PIL import Image
import pandas as pd

def rename_and_save(img_dir, csv):
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        prompt = csv.loc[csv['filename']==img]['prompt'].values[0]
        try:
            new_name = os.path.join(img_dir, f"{prompt}_000.png")
            os.rename(img_path, new_name)
        except:
            # continue
            os.remove(img_path)
    return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        required=True,

    )
    # parser.add_argument(
    #     "--out_dir",
    #     type=str,
    #     default=None,
    #     required=True,
    # )
    args = parser.parse_args()
    csv = pd.read_csv(args.csv_path)
    rename_and_save(args.img_dir, csv)

    

    
