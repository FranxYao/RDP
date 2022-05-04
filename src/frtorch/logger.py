import numpy as np
import sys
# from torch.utils.tensorboard import SummaryWriter

class TrainingLog(object):
  def __init__(self, model_name, output_path, 
    tensorboard_path=None, log_info=[], print_var=False, use_tensorboard=False):
    self.model_name = model_name
    self.output_path = output_path
    self.tensorboard_path = tensorboard_path
    self.print_var = print_var
    
    if(use_tensorboard):
      self.summary_writer = SummaryWriter(log_dir=self.tensorboard_path)
    else: self.summary_writer = None

    self.log = {}
    for n in log_info:
      self.log[n] = []
    return 

  def update(self, out_dict):
    """Update the log"""
    for l in out_dict: 
      if(l in self.log): self.log[l].append(out_dict[l])
      elif(isinstance(out_dict[l], float) or isinstance(out_dict[l], int)): 
        self.log[l] = [out_dict[l]]
    return

  def print(self):
    """Print out the log"""
    s = ""
    if(self.print_var):
      for l in self.log: s += "%s: mean = %.4g, var = %.4g " %\
        (l, np.average(self.log[l]), np.var(self.log[l]))
    else:
      for l in self.log: s += "%s: %.4g, " %\
        (l, np.average(self.log[l]))
    print(s)
    print("")
    return 

  def write(self, ei, log_metrics=None):
    """Write the log for current epoch"""
    log_path = self.output_path + "epoch_%d.log" % ei
    print("Writing epoch log to %s ... " % log_path)
    with open(log_path, "w") as fd:
      log_len = len(self.log[list(self.log.keys())[0]])
      for i in range(log_len):
        for m in self.log:
          if(log_metrics is None): 
            fd.write("%s: %.4g\t" % (m, self.log[m][i]))
          else:
            if(m in log_metrics): fd.write("%s: %.4f\t" % (m, self.log[m][i]))
        fd.write("\n")
    return 

  def reset(self):
    """Reset the log"""
    for l in self.log: 
      self.log[l] = []
    return

  def write_tensorboard(self, out_dict, n_iter, ei, mode, key=None):
    if(mode != 'train'): n_iter = ei
    if(key is None):
      for metrics in out_dict:
        if(metrics in self.log):
          self.summary_writer.add_scalar('%s/%s' % (mode, metrics), 
            out_dict[metrics], n_iter)
    else:
      self.summary_writer.add_scalar('%s/%s' % (mode, key), 
        out_dict[key], n_iter)
    return


class PrintLog(object):
  def __init__(self, log_path):
    self.terminal = sys.stdout
    self.log = open(log_path, "a", buffering=1)

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass    