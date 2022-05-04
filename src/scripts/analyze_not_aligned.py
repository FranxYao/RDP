# filename = '/home/s1946695/RDP/outputs/20220428/bertnet_0.0.6.4/bertnet_dev_epoch_-1_not_aligned.txt'
filename = '/home/s1946695/RDP/outputs/20220428/bertnet_0.1.9.1/bertnet_dev_epoch_-1_not_aligned.txt'

lines = open(filename).readlines()

total_cnt = 0
lex_cnt = 0
syn_cnt = 0
sem_cnt = 0
na_cnt = 0
for li, l in enumerate(lines):
  if(li % 3 == 0):
    l = l.split()
    total_cnt += int(l[3])
    if(l[-1] == 'LEX'): lex_cnt += int(l[3])
    if(l[-1] == 'SYN'): syn_cnt += int(l[3])
    if(l[-1] == 'SEM'): sem_cnt += int(l[3])
    if(l[-1] == 'NA'): na_cnt += int(l[3])

print(lex_cnt, syn_cnt, sem_cnt, na_cnt)

print('lex %.4f' % (lex_cnt / float(total_cnt)))
print('syn %.4f' % (syn_cnt / float(total_cnt)))
print('sem %.4f' % (sem_cnt / float(total_cnt)))
print('na %.4f' % (na_cnt / float(total_cnt)))
