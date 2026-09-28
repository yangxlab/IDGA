import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import re


Image_key={'CUHK-PEDES':'file_path','RSTPReid':'img_path','ICFG-PEDES':'file_path'}
class pair_dataset(Dataset):#加载图像和受干扰的文本
    def __init__(self, datasets,ann_file, test_transforms, image_root,adv_texts_path, max_words=77):  #原先max_words是30
        self.ann = json.load(open(ann_file, 'r'))
        self.image_root = image_root
        self.max_words = max_words
        self.transform = test_transforms
        
        self.text = {}
        self.text['text_id']=[]
        self.text['text']=[]
        self.text['original_text']=[]
        
        self.image = {}
        self.image['image_id']=[]
        self.image['image']=[]
        
        self.adv_text={}
        self.adv_text['adv_text_id']=[]
        self.adv_text['adv_text']=[]
        
        image_key=Image_key[datasets]
        for i, ann in enumerate(self.ann):
            if ann['split'] == "test":
                self.image['image'].append(ann[image_key])
                self.image['image_id'].append(ann['id'])
                for j, caption in enumerate(ann['captions']):
                    self.text['original_text'].append(caption)
                    self.text['text_id'].append(ann['id'])
                    self.adv_text['adv_text_id'].append(ann['id'])

        adv_texts=json.load(open(adv_texts_path, 'r'))
        for i, item in enumerate(adv_texts):
            self.adv_text['adv_text'].append(item)

    def __len__(self):
        return len(self.image['image'])

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image['image'][index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        person_id = self.image['image_id'][index]
        group_adv_text = ""
        group_text = ""
        for i in range(len(self.text['original_text'])):
            if self.text['text_id'][i] == person_id:
                if len(group_adv_text)==0:
                    group_adv_text= self.adv_text['adv_text'][i]
                    group_text = self.text['original_text'][i]
                else:
                    group_adv_text= group_adv_text+'|'+self.adv_text['adv_text'][i]
                    group_text = group_text +'|' + self.text['original_text'][i]
            elif group_adv_text:
                break

        return image,group_adv_text, person_id,group_text,image_path

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])", 
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption