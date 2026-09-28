import logging
import os
import sys
import os.path as op
import numpy as np
from prettytable import PrettyTable
import math

def setup_logger(name, save_dir, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not op.exists(save_dir):
        print(f"{save_dir} is not exists, create given directory")
        os.makedirs(save_dir)
    fh = logging.FileHandler(os.path.join(save_dir, "attack_log.txt"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def logger_metrics(metric):
    original_text_metrics_sim=[]
    original_text_metrics_ppl=[]
    paraphrase_metrics_sim=[]
    paraphrase_metrics_ppl=[]
    metrics=[]
    metrics.append(metric)
    original_text_metrics_sim.append(metrics[0][0])
    original_text_metrics_ppl.append(metrics[0][1])
    paraphrase_metrics_sim.append(metrics[0][2])
    paraphrase_metrics_ppl.append(metrics[0][3])
        
    mean_original_text_sim=np.mean(original_text_metrics_sim)
    mean_original_text_ppl=np.mean(original_text_metrics_ppl)
    mean_paraphrase_sim=np.mean(paraphrase_metrics_sim)
    mean_paraphrase_ppl=np.mean(paraphrase_metrics_ppl)
    
    table = PrettyTable(["text", "ppl", "sim"])
    table.add_row(['before attack', mean_original_text_ppl, mean_original_text_sim])
    table.add_row(['after attack', mean_paraphrase_ppl, mean_paraphrase_sim])
    table.custom_format["ppl"] = lambda f, v: f"{v:.3f}"
    table.custom_format["sim"] = lambda f, v: f"{v:.3f}"
    logger=logging.getLogger("IDGA")
    logger.info('\n' + str(table))