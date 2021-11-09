import argparse
import numpy as np 

import pandas as pd 

from pandas import DataFrame
from frtorch import str2bool

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

def define_argument():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--file_name", default='', type=str)
  parser.add_argument(
    "--print_history", default=False, type=str2bool)
  return parser

def read_scores(start_line, lines):
  results = []
  names = ['epoch'] + lines[start_line + 1].split()
  for i in range(40):
    li = start_line + 2 + i
    r = lines[li][:-1]
    if(not r or r[0].isnumeric() == False): 
      print('WARNING: max epoch %d' % (i - 1))
      break
    r = r.split()
    
    if(int(r[0]) != i): 
      print('WARNING: max epoch %d' % (i - 1))
      break
    r_ = [float(ri) for ri in r]
    results.append(r_)
  results_ = {n: [] for n in names}
  for r in results:
    for n, ri in zip(names, r):
      results_[n].append(ri)
  results = results_
  results = DataFrame(results)
  return results

def main():
  parser = define_argument()
  args = parser.parse_args()
  
  with open(args.file_name) as fd:
  # with open(file_name) as fd:
    lines = fd.readlines()
    li_ = len(lines)
    l_test = -1
    l_dev = -1
    for li, l in enumerate(lines[::-1]):
      li_ -= 1
      if(l.startswith('history test scores:') and li > 3): 
        l_test = max(l_test, li_)

      if(l.startswith('history validation:') and li > 3): 
        l_dev = max(l_dev, li_)

  dev_results = read_scores(l_dev, lines)
  test_results = read_scores(l_test, lines)

  print('best argmax results')
  print('best ib4 at epoch', dev_results['am_ib4'].argmax())
  best_epoch = dev_results['am_ib4'].argmax()
    
  if(args.print_history):
    print('Dev scores:')
    print(dev_results[[
      'am_ib4', 'am_bleu_2', 'am_bleu_4', 'am_sb_2', 'am_sb_4']])
  print('Best score at epoch %d:' % best_epoch)
  print(dev_results[['am_ib4', 'am_bleu_2', 'am_bleu_4', 'am_sb_2', 'am_sb_4']].iloc[best_epoch])

  if(args.print_history):
    print('Test scores:')
    print(test_results[['am_ib4', 'am_bleu_2', 'am_bleu_4', 'am_sb_2', 'am_sb_2']])
  print('Best dev score at epoch %d:' % best_epoch)
  print(test_results[['am_ib4', 'am_bleu_2', 'am_bleu_4', 'am_sb_2', 'am_sb_4']].iloc[best_epoch])
  print('----------------------------\n\n')

  print('best sampling results')
  print('best ib4 at epoch', dev_results['sp_ib4'].argmax())
  best_epoch = dev_results['sp_ib4'].argmax()
  if(args.print_history):
    print('Dev scores:')
    print(dev_results[[
      'sp_ib4', 'sp_bleu_2', 'sp_bleu_4', 'sp_sb_2', 'sp_sb_4']])
  print('Best score at epoch %d:' % best_epoch)
  print(dev_results[['sp_ib4', 'sp_bleu_2', 'sp_bleu_4', 'sp_sb_2', 'sp_sb_4']].iloc[best_epoch])
  if(args.print_history):
    print('Test scores:')
    print(test_results[['sp_ib4', 'sp_bleu_2', 'sp_bleu_4', 'sp_sb_2', 'sp_sb_4']])
  print('Best dev score at epoch %d:' % best_epoch)
  print(test_results[['sp_ib4', 'sp_bleu_2', 'sp_bleu_4', 'sp_sb_2', 'sp_sb_4']].iloc[best_epoch])
  return 

if __name__ == '__main__':
  main()
