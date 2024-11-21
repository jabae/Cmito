import torch
import torch.nn as nn


class Model(nn.Module):
  """
  Model wrapper for training.
  """
  
  def __init__(self, model):

    super(Model, self).__init__()
    self.model = model

  def forward(self, sample):

    mask = self.model(sample)    
    pred = torch.sigmoid(mask)
    
    return pred

  def save(self, fpath):

    torch.save(self.model.state_dict(), fpath)

  def load(self, fpath):

    state_dict = torch.load(fpath, weights_only=False)
    
    self.model.load_state_dict(state_dict, strict=False)