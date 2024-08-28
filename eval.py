
from numpy import clip
from auto_eval import run_clip_eval, CLIPConfig, run_blip_eval, BLIPConfig
import argparse 

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


    
    args = parser.parse_args()

    clip_eval_config = CLIPConfig(input_dir=args.img_dir, csv_path=args.csv_path)
    blip_eval_config = BLIPConfig(input_dir=args.img_dir, csv_path=args.csv_path)

    clip_result = run_clip_eval(clip_eval_config)
    blip_result = run_blip_eval(blip_eval_config)
    print(clip_result)
    print(blip_result)
