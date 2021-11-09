import argparse
import numpy as np 

import pandas as pd 

from pandas import DataFrame

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

def define_argument():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--file_name", default='', type=str)
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
  results['p_ib4'] = 0.9 * results['p_bleu_4'] - 0.1 * results['p_self_bleu_4']
  results['p_ib3'] = 0.9 * results['p_bleu_3'] - 0.1 * results['p_self_bleu_3']
  results['p_ib2'] = 0.9 * results['p_bleu_2'] - 0.1 * results['p_self_bleu_2']
  return results

def main():
  parser = define_argument()
  args = parser.parse_args()
  
  with open(args.file_name) as fd:
  # with open(file_name) as fd:
    lines = fd.readlines()
    li_ = len(lines)
    for l in lines[::-1]:
      li_ -= 1
      if(l.startswith('history test scores:')): l_test = li_
      if(l.startswith('history validation:')): 
        l_dev = li_
        break

  dev_results = read_scores(l_dev, lines)
  test_results = read_scores(l_test, lines)

  print('best ib4 at epoch', dev_results['p_ib4'].argmax())
  best_epoch = dev_results['p_ib4'].argmax()
    
  print('Dev scores:')
  print(dev_results[[
    'p_ib4', 'p_bleu_2', 'p_bleu_4', 'p_self_bleu_2', 'p_self_bleu_2']])
  print('Best score at epoch %d:' % best_epoch)
  print(dev_results[['p_ib4', 'p_bleu_2', 'p_bleu_4', 'p_self_bleu_2', 
    'p_self_bleu_4']].iloc[best_epoch])
  print('Test scores:')
  print(test_results[['p_ib4', 'p_bleu_2', 'p_bleu_4', 'p_self_bleu_2', 
    'p_self_bleu_4']])
  print('Best dev score at epoch %d:' % best_epoch)
  print(test_results[['p_ib4', 'p_bleu_2', 'p_bleu_4', 'p_self_bleu_2', 
    'p_self_bleu_4']].iloc[best_epoch])
  return 

if __name__ == '__main__':
  main()
