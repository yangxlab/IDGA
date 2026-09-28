import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from transformers import BertForMaskedLM,AutoTokenizer

from datatools import utils

from attack import *
from attack.multimodalAttack import MultiModalAttacker
from torchvision import transforms

from datatools.caption_dataset import pair_dataset
from torchvision import transforms

import os.path as op
from datatools.logger import setup_logger
import logging
from attack.RR_attack import get_metrics

import copy

from vic_model.inference import inference_text,inference_image
import tqdm

#多线程
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datatools.eval import compute_similarity,evalue

from vic_model.create_model import create_model
from datatools.build import build_transforms

from attack.RR_attack import ATO_attack
from datatools.logger import logger_metrics

from vic_model.tokenization_bert import BertTokenizer
from datatools.caption_dataset import pre_caption
import imageio

def retrieval_eval(source_model, target_model, data_loader,dataset_name,temperature,epsilon,num_iters, device):
    source_model.float()
    source_model.eval()
    target_model.float()
    target_model.eval()

    image_attacker = ImageAttacker(epsilon / 255., preprocess=source_model.images_normalize, bounding=(0, 1), cls=args.cls)
    multi_attacker = MultiModalAttacker(source_model, image_attacker,temperature, device, cls=args.cls)

    print('Forward')
    
    output_image={}
    output_text={}
    output_image_adv={}
    output_text_adv={}
    
    caption_pids= []
    images_pids= []
    
    original_text=[]
    adv_text=[]

    pbar = tqdm.tqdm(range(len(data_loader.dataset.image['image'])))
    id_lable=''
    adv_image_path = os.path.join(args.adv_images_path, source_model.model_name,"IDGA", args.dataset_name,"imgs")
    Flag=False
    for item in data_loader:
        image_path=item[4]
        images=item[0]
        original_images=copy.deepcopy(images)
        person_id=item[2]
        images_pids.append(person_id)
        group_adv_text=[]
        group_text=[]
        for adv_text_item,text_item in zip(item[1],item[3]):
            group_adv_text.append(adv_text_item.split('|'))
            group_text.append(text_item.split('|'))
        
        for i in range(len(group_text)):
            if id_lable==None or id_lable !=person_id[i]:
                id_lable=person_id[i]
                original_text.append(group_text[i])
                adv_text.append(group_adv_text[i])
                caption_pids.append(person_id[i].repeat(len(group_text[i])))
        #对图像实施改进的协同攻击策略
        images = multi_attacker.advanced_attack_image(images, group_adv_text, num_iters=num_iters)
        with torch.no_grad():
            output = inference_image(target_model,target_model.images_normalize(original_images))
            for key,value in output.items():
                if key not in output_image:
                    output_image[key] = []
                output_image[key].append(value.cpu().float().detach())
            
            output_adv = inference_image(target_model,target_model.images_normalize(images))
            for key,value in output_adv.items():
                if key not in output_image_adv:
                    output_image_adv[key] = []
                output_image_adv[key].append(value.cpu().float().detach())
            
        pbar.update(images.size(0))   
        
    for key,value in output_image_adv.items():
        if target_model.model_name == 'RaSa' and key == 'image_feat':
            output_image_adv[key] = (torch.cat(output_image_adv[key], dim=0))
        else:
            output_image_adv[key] = (torch.cat(output_image_adv[key], dim=0)).to(device)
         
    images_pids = torch.cat(images_pids, 0)
    caption_pids= torch.cat(caption_pids, 0)
    
    temp_original_text=[]
    for i in range(len(original_text)):
        for j in range(len(original_text[i])):
            temp_original_text.append(original_text[i][j])
    original_text=temp_original_text
       
    temp_adv_text=[]
    for i in range(len(adv_text)):
        for j in range(len(adv_text[i])):
            temp_adv_text.append(adv_text[i][j])
    adv_text=temp_adv_text
    
    result={}
    result['data']=[]
    for i in range(0,len(adv_text),64):
        sub_original_text=original_text[i:i+64]
        sub_adv_text=adv_text[i:i+64]
        with torch.no_grad():
            output = inference_text(target_model,sub_original_text)
            for key,value in output.items():
                if key not in output_text:
                    output_text[key] = []
                output_text[key].append(value.cpu().float().detach())
        
            output_adv = inference_text(target_model,sub_adv_text)
            for key,value in output_adv.items():
                if key not in output_text_adv:
                    output_text_adv[key] = []
                output_text_adv[key].append(value.cpu().float().detach())
            
        for j in range(len(sub_adv_text)):
            temp={}
            temp['text0']=sub_original_text[j]
            temp['text0_paraphrases']=[]
            temp['text0_paraphrases'].append(sub_adv_text[j])
            result['data'].append(temp)
    
    for key,value in output_text.items():
        output_text[key] = (torch.cat(output_text[key], dim=0)).to(device)
        
    for key,value in output_text_adv.items():
        output_text_adv[key] = (torch.cat(output_text_adv[key], dim=0)).to(device)

    metric=get_metrics(result,dataset_name,source_model.model_name)
    logger_metrics(metric)
    logger = logging.getLogger("IDGA")
    logger.info("Enter inferencing")
    
    print('before attack')
    similarity=compute_similarity(target_model,output_image,output_text)
    """ if target_model.model_name=='RaSa' or target_model.model_name=='APTM':
        dist.barrier()
        torch.distributed.all_reduce(similarity, op=torch.distributed.ReduceOp.SUM) """
    evalue(similarity, images_pids.cpu(), caption_pids.cpu())

    print('IDGA attack success rate:')
    similarity_adv=compute_similarity(target_model,output_image_adv,output_text_adv)
    """ if target_model.model_name=='RaSa' or target_model.model_name=='APTM':
        dist.barrier()
        torch.distributed.all_reduce(similarity_adv, op=torch.distributed.ReduceOp.SUM) """
    evalue(similarity_adv, images_pids.cpu(), caption_pids.cpu())

def main(args, source_model_config,target_model_config):
    # ---------- 1. 分布式初始化 ----------
    """ dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    args.gpu = device """          
    
    device = args.gpu

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
        
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    #### Model ####
    print("Creating Source Model")
    source_model=create_model(args.source_model_name,args.text_encoder,source_model_config,args.source_checkpointer_path,args.dataset_name).to(device)
    total_params = sum(p.numel() for p in source_model.parameters())
    print(f"Total number of parameters: {total_params}")
    print("Creating target model")
    target_model=create_model(args.target_model_name,args.text_encoder,target_model_config,args.target_checkpointer_path,args.dataset_name).to(device)


    #### Dataset ####
    print("Generating adv dataset")
    adv_texts_path=ATO_attack(args.dataset_name,tokenizer,source_model,args.cls,device)
    
    print("Creating dataset")
    transform = build_transforms(args.source_model_name)
    
    test_dataset=pair_dataset(args.dataset_name,args.test_file, transform, args.image_root,adv_texts_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)


    print("Start eval")
    start_time = time.time()

    retrieval_eval(source_model,target_model, test_loader,args.dataset_name,args.temperature,args.epsilon,args.num_iters, device)
    end = time.time()
    print("Time: ", end - start_time)
    print("game over!!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='RSTPReid')
    parser.add_argument('--test_file', default='export/RSTPReid/data_captions.json')
    parser.add_argument('--image_root', default='export/RSTPReid/imgs/')
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--temperature', default=0.004)
    parser.add_argument('--epsilon', default=2)
    parser.add_argument('--num_iters', default=10)
    parser.add_argument('--source_model_name', default='IRRA')
    parser.add_argument('--source_model_config', default='configs/IRRA/RSTPReid/config.yaml')
    parser.add_argument('--source_checkpointer_path', default='checkpoints/RSTPReid/IRRA/best.pth')
    parser.add_argument('--target_model_name', default='IRRA')
    parser.add_argument('--target_model_config', default='configs/IRRA/RSTPReid/config.yaml')
    parser.add_argument('--target_checkpointer_path', default='checkpoints/RSTPReid/IRRA/best.pth')
    parser.add_argument('--text_encoder', default='export/bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+', default=2)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--save_dir', default='log/RSTPReid')
    parser.add_argument('--adv_images_path', default='adv_images')

    args = parser.parse_args()

    args.cls = False
    source_model_config=yaml.load(open(args.source_model_config, 'r'), Loader=yaml.Loader)
    target_model_config=yaml.load(open(args.target_model_config, 'r'), Loader=yaml.Loader)

    logger = setup_logger('IDGA', save_dir=args.save_dir)
    logger.info('source model: '+ args.source_model_name)
    logger.info('target model: '+ args.target_model_name)
    main(args, source_model_config,target_model_config)

