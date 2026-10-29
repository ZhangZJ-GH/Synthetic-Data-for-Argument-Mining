import time
import numpy as np
import os, sys
import argparse
import datetime
from os.path import join
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--model", nargs='+',type=str,)
parsed_args = parser.parse_args()

gpu_id = parsed_args.gpu_id


map_id = {0:0, 1:1, 2:2, 3:3}


model = parsed_args.model
input_file = "./Struct-Syn/Data/cdcp/syn_datas/imitation_agent_100percent.jsonl"
for name in ["aaec_essay","cdcp","abstrct",]:
    if name in model:
        dataset_name=name
        break
for num in [5,100]:
    if f"{num}percent" in model:
        few_shot=num
        break
while True:
    os.system(f'nvidia-smi -q -d Memory |grep -A7 GPU|grep Free >tmp_{gpu_id}')
    memory_gpu = [int(x.split()[2]) for x in open(f'tmp_{gpu_id}', 'r').readlines()]
    os.system(f'rm tmp_{gpu_id}')
    print(memory_gpu)

    if memory_gpu[map_id[gpu_id]] > 7000:
        # gpu_id = str(np.argmax(memory_gpu))
        print(gpu_id)

        # os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))

        predict_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./Struct-Syn/graph_parser/log/distill_pretrain/{dataset_name}_{few_shot}percent/" + f"{predict_time}"
        prediction_log_dir = output_dir + "/prediction"
		
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m amparse.predictor.predict " \
                    f"--models {model} " \
                    f"--input {input_file} " \
                    f"--log {prediction_log_dir} " \
                    f"--batch_size 16"

        print(command)
        os.system(command)
        break

    time.sleep(10)
