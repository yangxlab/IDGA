from . import utils
import torch,gc
from prettytable import PrettyTable
import logging
def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        similarity = similarity.cpu()   
        indices = torch.argsort(similarity, dim=1, descending=True) 
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k shape:(2000,1000)
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k q_pids.view(-1, 1)表示将多维张量转为一维列向量

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum matches[:, :max_rank]表示取所有行，取前max_rank列
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100 


    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q matches (2000,1000)
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices
def rasa_similarity(sims_matrix, len_texts, len_images,image,text,model):
    device=next(model.parameters()).device
    score_matrix_t2i = torch.full((len_texts,len_images), -100.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    """ num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step) """
    start = 0
    end = len_texts
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    image_feats=image['image_feat']
    text_feats=text['text_feat']
    text_atts=text['text_atts']
    with torch.no_grad():
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=128, dim=0)
            topk_idx=topk_idx.to(image_feats.device)
            encoder_output = image_feats[topk_idx]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder.bert(encoder_embeds=text_feats[start + i].unsqueeze(0).expand(128, -1, -1).to(device),
                                            attention_mask=text_atts[start + i].unsqueeze(0).expand(128, -1).to(device),
                                            encoder_hidden_states=encoder_output.to(device),
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True,
                                            mode='fusion'
                                            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score
        del output
        torch.cuda.empty_cache()
    return score_matrix_t2i

def aptm_similarity(sims_matrix, len_texts, len_images,image,text,model):
    score_sim_t2i=sims_matrix
    device=next(model.parameters()).device
    score_matrix_t2i = torch.full((len_texts,len_images),  1000.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    """ num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step) """
    start = 0
    end = len_texts
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    image_embeds=image['image_feat']
    text_embeds=text['text_feat']
    text_atts=text['text_atts']
    with torch.no_grad():
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=128, dim=0)
            encoder_output = image_embeds[topk_idx]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.get_cross_embeds(encoder_output, encoder_att,
                                            text_embeds=text_embeds[start + i].repeat(128, 1, 1),
                                            text_atts=text_atts[start + i].repeat(128, 1))[:, 0, :]
            score = model.itm_head(output)[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score
            score_sim_t2i[start + i, topk_idx] = topk_sim
            score = model.itm_head(output)[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score
            score_sim_t2i[start + i, topk_idx] = topk_sim

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002*score_sim_t2i
    return score_matrix_t2i

def evalue(similarity, gids, qids):

    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
    table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

    # table.float_format = '.4'
    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
    logger=logging.getLogger("IDGA")
    logger.info('\n' + str(table))
        
    return t2i_cmc[0]

def compute_similarity(model,image,text):
    similarity=text['text_embed']@image['image_embed'].t()
    if model.model_name=='RaSa':
        similarity=rasa_similarity(similarity, len(text['text_embed']), len(image['image_embed']),image,text, model)
    elif model.model_name=='APTM':
        similarity=aptm_similarity(similarity, len(text['text_embed']), len(image['image_embed']),image,text, model)
    return similarity#6156 3074
    