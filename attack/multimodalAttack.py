import torch
import torch.nn.functional as F
from vic_model.inference import inference_text,inference_image
def equal_normalize(x):
    return x
class MultiModalAttacker():
    def __init__(self, net, image_attacker, temperature, device,cls=True):
        self.net = net
        self.image_attacker = image_attacker
        self.cls = cls
        self.device = device
        self.temperature=temperature
        if hasattr(image_attacker, 'preprocess'):
            self.image_normalize = image_attacker.preprocess
        else:
            self.image_normalize = equal_normalize

        self.repeat = 1
        if hasattr(image_attacker, 'repeat'):
            self.repeat = image_attacker.repeat
    def PGD(self,images,num_iters):
        images_adv = self.image_attacker.run_trades(self.net, images, num_iters)
        return images_adv
    def advanced_attack_image(self,images, adv_text, num_iters=10):
        temperature=self.temperature
        image_attack = self.image_attacker.attack(images, num_iters)
        
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        with torch.no_grad():
            text_adv_output=[]
            text_adv_embed=[]
            for i in range(len(adv_text)):
                """ text_input = self.tokenizer(adv_text[i], padding='max_length', truncation=True, max_length=77,
                                                return_tensors="pt").to(images.device) """
                text_adv_output.append(inference_text(self.net,adv_text[i]))
                #text_adv_output.append(self.net.inference_text(self.tokenizer,text_input))
                if self.cls:
                    text_adv_embed.append(text_adv_output[-1]['text_embed'][:, 0, :].detach())
                else:
                    text_adv_embed.append(text_adv_output[-1]['text_embed'].flatten(1).detach())

        with torch.no_grad():
            image_output = inference_image(self.net,self.image_normalize(images))
            #image_output = inference_image(self.net, self.image_normalize(images))
            if self.cls:
                image_embed = image_output['image_embed'][:, 0, :].detach()
            else:
                image_embed = image_output['image_embed'].flatten(1).detach()

        for i in range(num_iters):
            temp_text_adv_embed=[None]*image_embed.size(0)
            image_adv = next(image_attack)
            image_adv_output = inference_image(self.net,image_adv)
            #image_adv_output = inference_image(self.net,image_adv)
            if self.cls:
                image_adv_embed = image_adv_output['image_embed'][:, 0, :]
            else:
                image_adv_embed = image_adv_output['image_embed'].flatten(1)

            loss=torch.zeros([1]).to(image_embed.device)
            for index in range(image_embed.size(0)):
                temp_text_adv_embed[index]=torch.cat((text_adv_embed[index],image_embed[index].repeat(1,1)),dim=0)
            for index in range(image_adv_embed.size(0)):
                cos_score=torch.zeros([temp_text_adv_embed[index].size(0)]).to(image_embed.device)
                true_score=(torch.ones([temp_text_adv_embed[index].size(0)])/(temp_text_adv_embed[index].size(0))).to(image_embed.device)
                for j in range(temp_text_adv_embed[index].size(0)):
                    cos_score[j]=(image_adv_embed[index] @ temp_text_adv_embed[index][j].t())
                loss =loss + criterion((F.normalize(cos_score, dim=-1)/temperature).log_softmax(dim=-1),
                                          (F.normalize(true_score, dim=-1)/temperature).softmax(dim=-1).repeat(self.repeat, 1)) 

            loss =loss / (image_adv_embed.size(0)+1e-10)       
            loss.backward()
        

        images_adv = next(image_attack)
        return images_adv
