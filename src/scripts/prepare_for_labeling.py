# filename = '/home/s1946695/RDP/outputs/20220428/bertnet_0.0.6.4/bertnet_dev_epoch_-1_not_aligned.txt'
# outname = 'output1.txt'

filename = '/home/s1946695/RDP/outputs/20220428/bertnet_0.1.9.1/bertnet_dev_epoch_-1_not_aligned.txt'
outname = 'output2.txt'

lines = open(filename).readlines()
fd = open(outname, 'w')
for li, l in enumerate(lines):
  if(li % 3 == 0):
    l_ = l.split()
    if(l_[-1] not in ['LEX', 'SEM', 'SYN', 'NA']): 
      break
    fd.write(' '.join(l_[:-1]) + ' type=\n')
  else:
    fd.write(l)