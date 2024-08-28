import cv2 
import os 
import numpy as np
import sys


src = './attn_vis/' 

# token_idx = int(sys.argv[1]) if len(sys.argv) == 2 else None

token_idxs = sys.argv[1:]
token_idxs = [int(i) for i in token_idxs]

if len(token_idxs) == 0:
    raise

column = 8 
# 
# row = 16
row = len(os.listdir(os.path.join('attn_map', '0')))

width = row * 266
height = column * 266

width = (width+20) * len(token_idxs)

print(column, row)


out = cv2.VideoWriter('attn.avi', cv2.VideoWriter_fourcc(*"MJPG"), 1, (width,height))
for time_step in range(len(os.listdir(src))):

    long_frame = list()
    for token_idx in token_idxs:
        token_attn_dir = os.path.join(src, str(time_step), str(token_idx))
        num_attn_map = len(os.listdir(token_attn_dir))

        frame = list()
        for idx in range(column):
            rows = list()
            for jdx in range(row):
                attn_dir = os.path.join(token_attn_dir, f"{str(jdx)}_{str(idx)}.jpg")
                image = cv2.imread(attn_dir)
                # if os.path.exists(attn_dir):
                #     image = cv2.imread(attn_dir)
                # else:
                #     image = np.zeros((256, 256, 3), dtype=np.uint8)


                image = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, (25, 25, 25))
                rows.append(image) 
            rows = np.concatenate(rows, axis=1) 
            frame.append(rows) 
        frame = np.concatenate(frame, axis=0) 
        frame = cv2.copyMakeBorder(frame, 0, 0, 10, 10, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        long_frame.append(frame)
    long_frame = np.concatenate(long_frame, axis=1)
    out.write(long_frame) 
out.release()



    


