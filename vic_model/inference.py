import torch.nn.functional as F
from .IRRA.utils.simple_tokenizer import SimpleTokenizer
from .tokenization_bert import tokenize
import torch
from datatools.build import build_transforms_infer
from datatools.caption_dataset import pre_caption
from vic_model.APTM.dataset.utils import aptm_pre_caption
def inference_text(model,data):
    text={}
    device=next(model.parameters()).device
    if model.model_name == 'IRRA':
        tokenizer=SimpleTokenizer()
        text_token=[]
        for i in range(len(data)):
            text_token.append((tokenize(data[i], tokenizer=tokenizer, text_length=77, truncate=True)).tolist())
        text_input = torch.tensor(text_token).to(device)
        text['text_embed']=F.normalize(model.encode_text(text_input), dim=-1)
        #return {'text_embed': F.normalize(model.encode_text(text_input), dim=-1)}
    elif model.model_name == 'RaSa':
        RaSa_data=[]
        for i in range(len(data)):
            RaSa_data.append(pre_caption(data[i], 77))
        text_input = model.tokenizer(RaSa_data, padding='max_length', truncation=True, max_length=77, return_tensors="pt").to(device)
        text['text_atts']=text_input.attention_mask
        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text['text_feat'] = text_output.last_hidden_state
        text['text_embed'] = F.normalize(model.text_proj(text['text_feat'][:, 0, :]))
    elif model.model_name == 'APTM':
        APTM_data=[]
        for i in range(len(data)):
            #APTM_data.append(pre_caption(data[i], 77))
            APTM_data.append(aptm_pre_caption(data[i], 77, icfg_rstp=True))#对于所有数据集来说都是true
        text_input = model.tokenizer(APTM_data, padding='max_length', truncation=True, max_length=77,return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.text_proj(text_embed[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        text['text_feat']=text_embed
        text['text_atts']=text_input.attention_mask
        text['text_embed']=text_feat
    return text
    
def inference_image(model,data):
    image={}
    transform = build_transforms_infer(model.model_name)
    data=transform(data)
    device = next(model.parameters()).device
    data = data.to(device)
    if model.model_name == 'IRRA':
        image['image_embed']=F.normalize(model.encode_image(data), dim=-1)
    elif model.model_name == 'RaSa':
        image['image_feat'] = model.visual_encoder(data)
        image['image_embed'] = model.vision_proj(image['image_feat'][:, 0, :])
        image['image_embed'] = F.normalize(image['image_embed'], dim=-1)
    elif model.model_name == 'APTM':
        image_embed, _ = model.get_vision_embeds(data)
        image_feat = model.vision_proj(image_embed[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)
        image['image_feat']=image_embed
        image['image_embed']=image_feat
    return image