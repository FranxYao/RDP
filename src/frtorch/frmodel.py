class FRModel(object):
  """Fast research-level engineering with Torch, base model"""

  def __init__(self):
    self.model = ...  # to be instantialized with a torch model 
    self.optimizer = ...  # to be instantialized with a torch optimizer
    self.log_info = [] # to be specified
    self.validation_scores = ... # to be specified
    self.validation_criteria = ... # to be specified 
    return 

  def train(self):
    self.model.train()
    return 

  def eval(self):
    self.model.eval()
    return 

  def to(self, device):
    self.model.to(device)
    return 

  def parameters(self):
    return self.model.parameters()

  def named_parameters(self):
    return self.model.named_parameters()

  def state_dict(self):
    return self.model.state_dict()

  def load_state_dict(self, ckpt):
    self.model.load_state_dict(ckpt['model_state_dict'])
    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return 

  def zero_grad(self):
    self.model.zero_grad()
    return 

  def build(self):
    return 

  def train_step(self, batch_dict, n_iter, ei, bi):
    """
    Args:
      batch_dict:
      n_iter:
      ei:
      bi:
    """
    return 

  def inspect_step(self, batch_dict, n_iter, ei, bi):
    """Single step inspection during training
    
    Args:
      batch_dict:
      n_iter:
      ei:
      bi:
    """
    return 

  def val_step(self, batch_dict, n_iter, ei, bi):
    """
    Args:
      batch_dict:
      n_iter:
      ei:
      bi:
    """
    return 

  def test_step(self, batch_dict, n_iter, ei, bi):
    """
    Args:
      batch_dict:
      n_iter:
      ei:
      bi:
    """
    return 
    
  def val_end(self, outputs, n_iter, ei, bi, dataset, mode, output_path_base):
    return 