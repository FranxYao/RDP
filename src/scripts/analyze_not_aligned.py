# filename = '../annotation/yaofu/output1.txt'
filename = '../annotation/yaofu/output2.txt'

lines = open(filename).readlines()

total_cnt = 0
nlex = 0
lex_cnt = 0
nsyn = 0
syn_cnt = 0
nsem = 0
sem_cnt = 0
nna = 0
na_cnt = 0
for li, l in enumerate(lines):
  if(li % 3 == 0):
    l = l.split()
    total_cnt += int(l[3])
    if(l[-1].split('=')[-1] == 'LEX'): 
      nlex += 1
      lex_cnt += int(l[3])
    if(l[-1].split('=')[-1] == 'SYN'): 
      nsyn += 1
      syn_cnt += int(l[3])
    if(l[-1].split('=')[-1] == 'SEM'): 
      nsem += 1
      sem_cnt += int(l[3])
    if(l[-1].split('=')[-1] == 'NA'): 
      nna += 1
      na_cnt += int(l[3])

print(lex_cnt, syn_cnt, sem_cnt, na_cnt)
print(nlex, nsyn, nsem, nna)
print('lex %.4f' % (lex_cnt / float(total_cnt)))
print('syn %.4f' % (syn_cnt / float(total_cnt)))
print('sem %.4f' % (sem_cnt / float(total_cnt)))
print('na %.4f' % (na_cnt / float(total_cnt)))
