#!/bin/bash
python auto_run.py --scale 300 --preserve_scale 10 --lse_scale 1 --align_scale 1 --beta 0.9 --prompt_file ./dataset_prompts/CC-500.txt  --output_dir output/  --exp_name repro_cc500_256_sd15 --target_res 256 --model_path runwayml/stable-diffusion-v1-5
