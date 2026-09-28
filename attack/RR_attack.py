from fibber.benchmark.benchmark_adversarial_attack import Benchmark,get_strategy
import json
import torch
import os

from fibber.metrics.metric_utils import MetricBundle
from fibber.datasets.dataset_utils import get_dataset
from fibber.metrics.bert_lm_utils import get_lm
from fibber.paraphrase_strategies.asrs_utils_wpe import get_wordpiece_emb
from torch.utils.data import DataLoader

from vic_model.inference import inference_text
DATASET_CONFIG = {
    "RSTPReid":{
        "dataset": "RSTPReid",
        "output_dir": "exp-RSTPReid",
    },
    "CUHK-PEDES":{
        "dataset": "CUHK-PEDES",
        "output_dir": "exp-CUHK-PEDES",
    },
    "ICFG-PEDES":{
        "dataset": "ICFG-PEDES",
        "output_dir": "exp-ICFG-PEDES",
    },
}
GPU_CONFIG = {
    "cpu": {
        "--bert_ppl_gpu_id": -1,
        "--use_gpu_id": -1,
        "--strategy_gpu_id": -1,
    },
    "gpu": {
        "bert_ppl_gpu_id": 0,
        "use_gpu_id": 0,
        "strategy_gpu_id": 0,
    },
}
STRATEGY_CONFIG = {
    "rr": {
        "strategy": "RewriteRollbackStrategy",
        "rr_enforcing_dist": "wpe",
        "rr_wpe_threshold": 1.0,
        "rr_wpe_weight": 5,
        "rr_sim_threshold": 0.95,
        "rr_sim_weight": 15,
        "rr_ppl_weight": 5,
        "rr_sampling_steps": 50,
        "rr_reid_weight": 7,
        "rr_window_size": 3,
        "rr_accept_criteria": "joint_weighted_criteria",
        "rr_lm_option": "finetune",
        "rr_sim_metric": "USESimilarityMetric",
        "rr_early_stop": "half",
        #"rr_batch_size": 10,
        "rr_top_k": 100,
        "rr_temperature": 1.0,
        "rr_lm_steps": 5000,
    },
}
""" COMMON_CONFIG = {
    "--subsample_testset": 1000,
    "--max_paraphrases": 50,
    "--task": "attack",
    "--target_classifier": "transformer",
    "--transformer_clf_model_init": "bert-base"
}
DEFENSE_CONFIG = {
    "none": {},
    "lmag": {
        "--defense_strategy": "LMAgStrategy"
    },
    "sem": {
        "--defense_strategy": "SEMStrategy",
    },
    "adv": {
        "--defense_strategy": "AdvTrainStrategy",
    },
    "sapd": {
        "--defense_strategy": "SAPDStrategy",
    }
} """
class RRAttack():
    def __init__(self, gpu,dataset,exp_name,max_paraphrases,tokenizer,model,cls):
        self.gpu = GPU_CONFIG[gpu]
        self.dataset = DATASET_CONFIG[dataset]
        self.strategy= STRATEGY_CONFIG['rr']
        self.exp_name = exp_name
        self.max_paraphrases = max_paraphrases
        self.tokenizer = tokenizer
        self.cls=cls
        self.victim_model= model
        self.text_embed_mean = torch.Tensor()
    def get_text_embed_mean(self):
        path=os.path.join(os.getcwd(),'.fibber/datasets',self.dataset['dataset'],'test.json')
        ann = json.load(open(path, 'r'))
        texts= [item for item in ann['data']]

        index=0
        subtexts=[]
        for i,item in enumerate(texts):
            if(item['label']==texts[index]['label']):
                subtexts.append(item['text0'])
            else:
                #sub_text_inputs = self.tokenizer(subtexts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(self.victim_model.device)
                text_output=inference_text(self.victim_model,subtexts)
                with torch.no_grad():
                    sub_text_embed_mean=torch.mean(text_output['text_embed'],dim=0).cpu().float().detach()
                    self.text_embed_mean=torch.cat((self.text_embed_mean,sub_text_embed_mean.repeat(1,1)),dim=0)
                index=i
                subtexts.clear()
                subtexts.append(item['text0'])
        text_output=inference_text(self.victim_model,subtexts)
        with torch.no_grad():
            sub_text_embed_mean=torch.mean(text_output['text_embed'],dim=0).cpu().float().detach()
            self.text_embed_mean=torch.cat((self.text_embed_mean,sub_text_embed_mean.repeat(1,1)),dim=0)
        subtexts.clear()
    def generate_adversial_sample(self,save_path):
        benchmark = Benchmark(save_path,self.dataset["dataset"],subsample_attack_set=self.subsample_testset,use_gpu_id=self.gpu["use_gpu_id"],
                              bert_ppl_gpu_id=self.gpu["bert_ppl_gpu_id"],)
        
        # Get paraphrase strategy
        paraphrase_strategy = get_strategy(self.strategy,self.dataset["dataset"],self.strategy["strategy"],self.gpu["strategy_gpu_id"],self.dataset["output_dir"],benchmark.get_metric_bundle(), field='text0')
        
        benchmark.run_benchmark(victim_model=self.victim_model,
                                text_embed_mean=self.text_embed_mean,
                                paraphrase_strategy=paraphrase_strategy,
                                max_paraphrases=self.max_paraphrases,
                                exp_name=self.exp_name,
                                subsample_attack_set=self.subsample_testset)
    
    def gain_adv_texts(self,net,device,max_length=77):
        trainset, testset = get_dataset(self.dataset["dataset"])
        self.subsample_testset=len(testset['data'])
        filename = os.path.join(self.dataset["output_dir"], "adv_texts.json")
        path=os.path.join(os.getcwd(),'adv_texts',net.model_name,filename)
        if os.path.exists(path):
            return path
        exp_name = (self.dataset["dataset"] + "-" + self.strategy['strategy'] +"-"+ self.strategy['rr_sim_metric'] +"-"+str(self.subsample_testset))
        output_filename = os.path.join(self.dataset["output_dir"], exp_name + "-with-metric.json")
        output_path=os.path.join(os.getcwd(),'adv_texts',net.model_name,output_filename)
        
        if not os.path.exists(output_path):
            get_lm('finetune', self.dataset["dataset"], trainset, device,lm_steps=5000)
            get_lm('ppl', self.dataset["dataset"], trainset, device,lm_steps=5000)
            get_wordpiece_emb(self.dataset["dataset"], trainset, self.tokenizer, device)
            self.get_text_embed_mean()
            self.generate_adversial_sample(os.path.join(os.getcwd(),'adv_texts',net.model_name,self.dataset["output_dir"]))

        data_record = json.load(open(output_path, 'r'))
        #抉择对抗文本
        
        #trainset, testset = get_dataset(self.dataset["dataset"])#获取原样本
        order_adv_text={}
        order_adv_text['text0']=[]
        order_adv_text['text0_paraphrases']=[]
        order_adv_text['original_text_metrics']=[]
        order_adv_text['paraphrase_metrics']=[]

        final_adverse = []
        original_text_sim=[]
        original_text_ppl=[]
        paraphrase_sim=[]
        paraphrase_ppl=[]
        for i in range(len(testset['data'])):
            for ii,item in enumerate(data_record['data']):
                if item['text0'] == testset['data'][i]['text0']:#校正顺序扰乱的对抗样本
                    order_adv_text['text0'].append(item['text0'])
                    order_adv_text['text0_paraphrases'].append(item['text0_paraphrases'])
                    order_adv_text['original_text_metrics'].append(item['original_text_metrics'])
                    order_adv_text['paraphrase_metrics'].append(item['paraphrase_metrics'])
                    break
            if len(order_adv_text['text0'])==100 or i==len(testset['data'])-1:
                #order_text_inputs = self.tokenizer(order_adv_text['text0'], padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
                origin_output = inference_text(net,order_adv_text['text0']) 
                
                if self.cls:
                    origin_embeds = origin_output['text_embed'][:, 0, :].detach()
                else:
                    origin_embeds = origin_output['text_embed'].flatten(1).detach()

                criterion = torch.nn.KLDivLoss(reduction='none')
                for i in range(len(order_adv_text['text0'])):
                    #replace_text_input = self.tokenizer(order_adv_text['text0_paraphrases'][i], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
                    replace_output = inference_text(net,order_adv_text['text0_paraphrases'][i])
                    if self.cls:
                        replace_embeds = replace_output['text_embed'][:, 0, :]
                    else:
                        replace_embeds = replace_output['text_embed'].flatten(1)

                    loss = criterion(replace_embeds.log_softmax(dim=-1), origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1)) 
                    loss = loss.sum(dim=-1)
                    for j in range(len(order_adv_text['paraphrase_metrics'][i])): #尝试降低一些攻击成功率，提高文本的相似度,和流畅度
                        loss[j] = loss[j] + 0.015*order_adv_text['paraphrase_metrics'][i][j]['USESimilarityMetric']-0.00002*order_adv_text['paraphrase_metrics'][i][j]['BertPerplexityMetric']
                    candidate_idx = loss.argmax()

                    final_adverse.append(order_adv_text['text0_paraphrases'][i][candidate_idx])
                    original_text_sim.append(order_adv_text['original_text_metrics'][i]['USESimilarityMetric'])
                    original_text_ppl.append(order_adv_text['original_text_metrics'][i]['BertPerplexityMetric'])
                    paraphrase_sim.append(order_adv_text['paraphrase_metrics'][i][candidate_idx]['USESimilarityMetric'])
                    paraphrase_ppl.append(order_adv_text['paraphrase_metrics'][i][candidate_idx]['BertPerplexityMetric'])
                
                order_adv_text={}
                order_adv_text['text0']=[]
                order_adv_text['text0_paraphrases']=[]
                order_adv_text['original_text_metrics']=[]
                order_adv_text['paraphrase_metrics']=[] 
                   
        with open(path, 'w') as f:
            json.dump(final_adverse,f,indent=2)   
        return path     

         
def get_metrics(results,dataset_name,model_name):
    trainset, testset = get_dataset(dataset_name)
    metrics=MetricBundle(
            dataset_name=dataset_name,
            trainset=trainset, testset=testset,
            enable_use_similarity=True, use_gpu_id=0,
            enable_bert_perplexity=True, bert_ppl_gpu_id=0,
            field='text0'
        )
    output_filename = os.path.join(os.getcwd(),'adv_texts',model_name,'exp-'+dataset_name, 'adv_texts'+'-with-metric.json')
    result_metrics=metrics.measure_dataset(
            results=results, output_filename=output_filename)
    original_text_sim=[]
    original_text_ppl=[]
    paraphrase_sim=[]
    paraphrase_ppl=[]
    for i, item in enumerate(result_metrics['data']):
        original_text_sim.append(item['original_text_metrics']['USESimilarityMetric'])
        original_text_ppl.append(item['original_text_metrics']['BertPerplexityMetric'])
        paraphrase_sim.append(item['paraphrase_metrics'][0]['USESimilarityMetric'])
        paraphrase_ppl.append(item['paraphrase_metrics'][0]['BertPerplexityMetric'])
    return original_text_sim,original_text_ppl,paraphrase_sim,paraphrase_ppl

def ATO_attack(dataset_name,tokenizer,source_model,cls,device):
    text_attacker=RRAttack('gpu',dataset_name,None,10,tokenizer,source_model,cls=cls)#ICFG-PEDES:19848  CUHK-PEDES:6156
    print("gain adv_texts.....")
    adv_texts_path=text_attacker.gain_adv_texts(source_model,device)
    return adv_texts_path