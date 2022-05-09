import torch 
import copy
import os

import numpy as np 
import pandas as pd 

from frtorch import torch_model_utils as tmu
from frtorch import TrainingLog

from tqdm import tqdm
from pandas import DataFrame
from time import time
from pprint import pprint 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)


class Controller(object):
  """Controller for training, validation, and evaluation
  """

  def __init__(self, args, model, dataset):
    super(Controller, self).__init__()

    self.model_name = args.model_name
    self.model_version = args.model_version
    self.inspect_grad = args.inspect_grad
    self.inspect_model = args.inspect_model
    self.dataset = args.dataset
    self.task = args.task

    self.is_test = args.is_test
    self.fast_test_pipeline = args.fast_test_pipeline
    self.fast_train_pipeline = args.fast_train_pipeline
    self.use_tensorboard = args.use_tensorboard
    self.tensorboard_path = args.tensorboard_path

    self.write_output = args.write_output
    # self.write_output_full = args.write_output_full

    self.validation_scores = {}
    # for n in model.validation_scores:
    #   self.validation_scores[n] = []
    self.save_checkpoints = args.save_checkpoints

    self.num_epoch = args.num_epoch
    self.start_epoch = args.start_epoch 
    self.validate_start_epoch = args.validate_start_epoch
    self.print_log_per_nbatch = args.print_log_per_nbatch
    self.model_path = args.model_path
    self.output_path = args.output_path
    self.device = args.device
    self.batch_size = args.batch_size

    self.validation_criteria = args.validation_criteria

    # logging 
    self.logger = TrainingLog(self.model_name, self.output_path, 
      self.tensorboard_path, model.log_info, args.print_var, self.use_tensorboard) 
    return 

  def save_ckpt(self, model, ei):
    """Save the model after epoch"""
    # save_path = self.model_path + 'ckpt_e%d' % ei
    save_path = self.model_path  + 'ckpt'
    if(self.save_checkpoints == 'multiple'):
      save_path += '-e%d' % ei 
    save_path += '.pt'
    print('Epoch %d, saving the model at: %s' % (ei, save_path))
    if(hasattr(model, 'save')):
      model.save(save_path)
    else:
      torch.save(
        {'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': model.optimizer.state_dict()}, 
        save_path)
    return 


  def train(self, model, dataset):
    """Train the model
    
    TODO: integrate pytorch hyperparameter scheduler
    """
    print('Start training ... ')

    history_validation = dict()
    best_validation = -1e10
    best_validation_epoch = -1
    best_validation_scores = None
    history_test = {}

    start_time = time()

    train_dataloader = dataset.train_dataloader()
    num_batches = len(train_dataloader)
    print('train dataset, %d batches in total' % num_batches)

    # self.train_schedule_init(num_batches, self.start_epoch, self.num_epoch)
    if(self.fast_test_pipeline): 
      _, score = self.validate(model, dataset, -1, -1, 'test')
      print('test validate scores:')
      pprint(score)

    n_iter = self.start_epoch * num_batches - 1
    for ei in range(self.start_epoch, self.num_epoch):
      model.train()
      # before epoch 
      self.logger.reset()
      epoch_start_time = time()
      for bi, batch in enumerate(train_dataloader):
        # for n in batch: 
        #   if(isinstance(batch[n], torch.Tensor)):
        #     batch[n] = batch[n].to(self.device)
        n_iter += 1

        if(self.fast_train_pipeline):
          if(bi == 100): break

        out_dict = model.train_step(batch, n_iter, ei, bi)
        self.logger.update(out_dict)

        if(self.use_tensorboard):
          self.logger.write_tensorboard(out_dict, n_iter, ei, 'train')

        if(bi % self.print_log_per_nbatch == 0): 
          print(
            '\nmodel %s version %s; ' % 
              (self.model_name, self.model_version) + 
            'epoch %d/%d batch %d/%d n_iter %d; ' % 
              (ei, self.num_epoch, bi, num_batches, n_iter) + 
            'time %ds batch time %.2fs' % 
              (time() - start_time, (time() - epoch_start_time) / (bi + 1))
          )
          self.logger.print()

          if(self.inspect_grad == 'first'):
            tmu.print_grad(model) 
          elif(self.inspect_grad == 'second'):
            tmu.print_grad(model, level='second')

          if(self.inspect_model):
            inspect_dict = model.inspect_step(
              batch, out_dict, n_iter, ei, bi, dataset)
            # TODO: print inspect
            
        # if(bi % (self.print_log_per_nbatch // 5) == 0):
        #   print('.', end=' ', flush=True)

      # after epoch 
      print('model %s %s epoch %d finished, time: %d' % 
        (self.model_name, self.model_version, ei, time() - start_time))
      self.logger.print() 
      
      print('----------------------------------------------------------------')
      if(ei >= self.validate_start_epoch and self.fast_train_pipeline == False):
        validation_criteria, validation_scores = self.validate(
          model, dataset, ei, n_iter, 'dev')
        for s in validation_scores:
          if(s not in history_validation): 
            history_validation[s] = [validation_scores[s]]
          else:
            history_validation[s].append(validation_scores[s])

        if(validation_criteria is None or validation_criteria > best_validation):
          if(validation_criteria is None):
            print('Validation criteria not specified, save the model if required')
          else: 
            print('validation increase from %.4f to %.4f, save the model if required' %
              (best_validation, validation_criteria))
          print('current validation score:')
          print(DataFrame(validation_scores, index=['values']))
          best_validation = validation_criteria
          best_validation_epoch = ei
          best_validation_scores = copy.deepcopy(validation_scores)
          # save model 
          if(self.save_checkpoints != 'none'):
            self.save_ckpt(model, ei)
        else: 
          print(
            'Validation %.4f, no improvement, keep best at epoch %d' % 
            (validation_criteria, best_validation_epoch))
          print('current validation score:')
          print(DataFrame(validation_scores, index=['values']))
          print('best validation score:')
          print(DataFrame(best_validation_scores, index=['values']))
        print('history validation:')
        print(DataFrame(history_validation))
        # tmu.pprint_dict_nums(history_validation, index_list=True)
        print('----------------------------------------------------------------')
        print()

        # NOTE: currently skip testing
        # test_criteria, test_scores = self.validate(
        #   model, dataset, ei, n_iter, 'test')
        # for s in test_scores:
        #   if(s not in history_test): history_test[s] = [test_scores[s]]
        #   else: history_test[s].append(test_scores[s])
        # print('test scores:')
        # print(DataFrame(test_scores, index=['values']))
        # print('history test scores:')
        # print(DataFrame(history_test))
      else: 
        print('validate_start_epoch = %d, current %d, do not validate' % 
          (self.validate_start_epoch, ei))
    
    self.validate(model, dataset, -1, -1, 'test')
    return

  def validate(self, model, dataset, ei, n_iter, mode='dev'):
    """Validation

    Args:
      dataset: the dataset class 
      ei: number of epoch 
      n_iter: current iteration counter 
      mode: 'dev' or 'test'

    """
    # predictions visualization TBC 
    print('Model %s_%s, epoch %d, n_iter %d, validation / test on %s set ..' % 
      (self.model_name, self.model_version, ei, n_iter, mode))
    model.eval()

    output_path_base = self.output_path + self.model_name + '_' + mode

    inspect_at = np.random.randint(self.print_log_per_nbatch)

    batches, outputs = [], []
    scores = copy.deepcopy(self.validation_scores)

    if(mode == 'dev'): dataloader = dataset.val_dataloader()
    else: dataloader = dataset.test_dataloader()

    num_batches = len(dataloader)
    print('%d batches in total' % num_batches)

    start_time = time()
    for bi, batch in enumerate(dataloader):
      # for n in batch: 
      #   if(isinstance(batch[n], torch.Tensor)):
      #     batch[n] = batch[n].to(self.device)
      out_dict = model.val_step(batch, n_iter, ei, bi, dataset)
      outputs.append(out_dict)
      # batches.append(batch)
      for n in out_dict:
        if(n in scores): scores[n].append(out_dict[n])
        elif(isinstance(out_dict[n], float) or isinstance(out_dict[n], int)):
          scores[n] = [out_dict[n]]
      if(bi % 20 == 0): 
        print('.', end=' ', flush=True)
      # if(self.fast_test_pipeline and bi == 50): break

    print('')

    # import pdb; pdb.set_trace()
    val_end_scores = model.val_end(
      outputs, n_iter, ei, bi, dataset, mode, output_path_base)

    # average dev scores
    for n in scores: 
      if(len(scores[n]) != 0): scores[n] = float(np.average(scores[n]))
      else: scores[n] = -1

    if(val_end_scores):
      for n in val_end_scores: 
        if(n in scores):
          print('WARNING: %s already in batch validation out_dict! ')
          print('         overwrite with val_end_scores')
        scores[n] = val_end_scores[n]

    if(self.use_tensorboard):
      for n in scores:
        if(isinstance(scores[n], float)):
          self.logger.write_tensorboard(scores, n_iter, ei, mode, n)
          # tensorboard_writer.add_scalar(mode + '/' + n, scores[n], n_iter)

    print('')
    print('validation/ test finished, time: %.2f' % 
      (time() - start_time))

    model.train()
    if(self.validation_criteria != ''):
      return scores[self.validation_criteria], scores
    else:
      return None, scores

  def test(self, model, dataset):
    """TBC"""
    return 