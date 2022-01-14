from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    """
    Model wrapper for training.
    """
    def __init__(self, model, opt):
        super(Model, self).__init__()
        self.model = model
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec
        self.pretrain = opt.pretrain is not None
        self.mip = opt.mip

    def forward(self, sample):
        
        # Fold detection
        image = sample["image"]

        mask = self.model(image)
        
        preds = {}
        preds["mask"] = mask

        # Loss evaluation
        losses = self.eval_loss(preds, sample)
        
        return losses, preds

    def eval_loss(self, preds, sample):
        losses = dict()

        ## Discrim
        for k in self.out_spec:
            
            loss = F.binary_cross_entropy_with_logits(input=preds[k][0,0,32:-32,32:-32], target=sample[k][0,0,32:-32,32:-32])
            losses[k] = loss.unsqueeze(0)
            
        return losses

    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)

    def load(self, fpath):
        state_dict = torch.load(fpath)
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)
