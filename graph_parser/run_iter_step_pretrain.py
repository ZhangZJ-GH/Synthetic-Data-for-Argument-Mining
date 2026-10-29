import time
import numpy as np
import os, sys
import argparse
import datetime
from os.path import join
import json
def read_jsonl(file):
    datas = []
    with open(file, 'r') as f:
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas
def save_jsonl(datas, output_file):
    with open(output_file, "w") as fout:
        for data in datas:
            if data:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
def mix_sample(file1, GTfile2, output_file):
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    datas1 = read_jsonl(file1)
    datas2 = read_jsonl(GTfile2)
    new_datas = datas1 + datas2
    save_jsonl(new_datas, output_file)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int)
parser.add_argument("--start_seed", type=int)
parser.add_argument("--train1_file", type=str)
parser.add_argument("--train2_file", type=str)
parser.add_argument("--valid_file", type=str)
parser.add_argument("--test_file", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--epoch", type=int)
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--few_shot", type=int)
parsed_args = parser.parse_args()


gpu_id = parsed_args.gpu_id
start_seed = parsed_args.start_seed
train1_file = parsed_args.train1_file
train2_file = parsed_args.train2_file
valid_file = parsed_args.valid_file
test_file = parsed_args.test_file
log_dir = parsed_args.log_dir
epoch = parsed_args.epoch
model_path = parsed_args.model_path
dataset = parsed_args.dataset
few_shot = parsed_args.few_shot
trainning_log_dir = log_dir

map_id = {0:0, 1:1, 2:2, 3:3}
dataset_lr={"aaec_para":9.1e-5, "aaec_essay":9.1e-5, "cdcp":5.6e-5, "abstrct":8.1e-5}
dataset_lambda={"aaec_para":(1,0.18,1.05,0.21), "aaec_essay":(1,0.18,1.05,0.21), "cdcp":(1,0.057,0.82,0.15), "abstrct":(1,0.035,0.58,0.17)}
pretrain_lr=2e-5
pretrain_epoch=10
lambda_bio,lambda_proposition,lambda_arc,lambda_rel = dataset_lambda[dataset]
lr = dataset_lr[dataset]
if train1_file is None or train1_file == "None":
	command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m amparse.trainer.train " \
		f"--log {trainning_log_dir+'/sft'} " \
		f"--ftrain {train2_file} " \
		f"--fvalid {valid_file} " \
		f"--ftest {test_file} " \
		f"--seed {start_seed} " \
		f"--model_name_or_path {model_path} " \
		f"--build_numericalizer_on_entire_corpus true " \
		f"--batch_size 4 " \
		f"--eval_batch_size 16 " \
		f"--embed_dropout 0.1 " \
		f"--mlp_dropout 0.1 " \
		f"--dim_mlp 768 " \
		f"--dim_biaffine 768 " \
		f"--lambda_bio {lambda_bio} " \
		f"--lambda_proposition {lambda_proposition} " \
		f"--lambda_arc {lambda_arc} " \
		f"--lambda_rel {lambda_rel} " \
		f"--lr {lr} " \
		f"--warmup_ratio 0.1 " \
		f"--clip 5.0 " \
		f"--epochs {epoch} " \
		f"--terminate_epochs {epoch} " \
		f"--evaluate_epochs 1"
	print(command)
	os.system(command)
else:
	command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m amparse.trainer.train " \
		f"--log {trainning_log_dir+'/pretrain'} " \
		f"--ftrain {train1_file} " \
		f"--fvalid {valid_file} " \
		f"--ftest {test_file} " \
		f"--seed {start_seed} " \
		f"--model_name_or_path {model_path} " \
		f"--build_numericalizer_on_entire_corpus true " \
		f"--batch_size 4 " \
		f"--eval_batch_size 16 " \
		f"--embed_dropout 0.1 " \
		f"--mlp_dropout 0.1 " \
		f"--dim_mlp 768 " \
		f"--dim_biaffine 768 " \
		f"--lambda_bio {lambda_bio} " \
		f"--lambda_proposition {lambda_proposition} " \
		f"--lambda_arc {lambda_arc} " \
		f"--lambda_rel {lambda_rel} " \
		f"--lr {pretrain_lr} " \
		f"--warmup_ratio 0.1 " \
		f"--clip 5.0 " \
		f"--epochs {pretrain_epoch} " \
		f"--terminate_epochs {pretrain_epoch} " \
		f"--evaluate_epochs 1"
	print(command)
	os.system(command)

	command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m amparse.trainer.train " \
		f"--log {trainning_log_dir+'/sft'} " \
		f"--ftrain {train2_file} " \
		f"--fvalid {valid_file} " \
		f"--ftest {test_file} " \
		f"--seed {start_seed} " \
		f"--model_name_or_path {trainning_log_dir+'/pretrain/model'} " \
		f"--build_numericalizer_on_entire_corpus true " \
		f"--batch_size 4 " \
		f"--eval_batch_size 16 " \
		f"--embed_dropout 0.1 " \
		f"--mlp_dropout 0.1 " \
		f"--dim_mlp 768 " \
		f"--dim_biaffine 768 " \
		f"--lambda_bio {lambda_bio} " \
		f"--lambda_proposition {lambda_proposition} " \
		f"--lambda_arc {lambda_arc} " \
		f"--lambda_rel {lambda_rel} " \
		f"--lr {lr} " \
		f"--warmup_ratio 0.1 " \
		f"--clip 5.0 " \
		f"--epochs {epoch} " \
		f"--terminate_epochs {epoch} " \
		f"--evaluate_epochs 1"
	print(command)
	os.system(command)

