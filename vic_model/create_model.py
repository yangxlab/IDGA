import os.path as op
from easydict import EasyDict as edict
import torch

from .IRRA.utils.checkpoint import Checkpointer
from .IRRA.model import build_model

from .RaSa.models.model_person_search import ALBEF
from .RaSa.models.vit import interpolate_pos_embed
from .tokenization_bert import BertTokenizer
from torchvision import transforms

from .APTM.models.model_retrieval import APTM_Retrieval

num_classes_mapping = {
    'RSTPReid':3701,
    'CUHK-PEDES':11003,
    'ICFG-PEDES':3102
}
def create_model(model_name, text_encoder,config,checkpointer_path,dataset_name):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    if model_name == 'IRRA':
        model = build_model(edict(config), num_classes=num_classes_mapping[dataset_name])#num_classes=3102 是ICFG-PEDES数据集
        checkpointer = Checkpointer(model)
        checkpointer.load(checkpointer_path)
        model.tokenizer=tokenizer
        model.model_name=model_name
        model.images_normalize=transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    elif model_name == 'RaSa':
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(checkpointer_path, map_location='cpu')
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
        model.model_name=model_name
        model.images_normalize=transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    elif model_name == 'APTM':
        model = APTM_Retrieval(config=config)
        model.load_pretrained(checkpointer_path, config, is_eval=True)
        model.tokenizer=tokenizer
        model.model_name=model_name
        if dataset_name == 'RSTPReid':
            model.images_normalize=transforms.Normalize((0.27722597, 0.26065794, 0.3036557), (0.2609547, 0.2508087, 0.25293276))
        elif dataset_name == 'CUHK-PEDES':
            model.images_normalize=transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555))
        elif dataset_name == 'ICFG-PEDES':
            model.images_normalize=transforms.Normalize((0.30941582, 0.28956893, 0.30347288), (0.25849792, 0.24547698, 0.2366199))
    return model