import numpy as np

filenames = ['../annotation/yaofu/output1.txt', 
             '../annotation/yaofu/output2.txt',
             '../annotation/litu/output1.txt', 
             '../annotation/litu/output2.txt', 
             '../annotation/jinyuan/output1.txt', 
             '../annotation/jinyuan/output2.txt', 
            ]
lex_avg_out1, lex_avg_out2 = [], []
nlex_avg_out1, nlex_avg_out2 = [], []
syn_avg_out1, syn_avg_out2 = [], []
nsyn_avg_out1, nsyn_avg_out2 = [], []
sem_avg_out1, sem_avg_out2 = [], []
nsem_avg_out1, nsem_avg_out2 = [], []
na_avg_out1, na_avg_out2 = [], []
nna_avg_out1, nna_avg_out2 = [], []
for filename in filenames: 
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

    if(filename.split('.')[-2][-1] == '1'):
      lex_avg_out1.append(lex_cnt)
      nlex_avg_out1.append(nlex)
      syn_avg_out1.append(syn_cnt)
      nsyn_avg_out1.append(nsyn)
      sem_avg_out1.append(sem_cnt)
      nsem_avg_out1.append(nsem)
      na_avg_out1.append(na_cnt)
      nna_avg_out1.append(nna)
    else:
      lex_avg_out2.append(lex_cnt)
      nlex_avg_out2.append(nlex)
      syn_avg_out2.append(syn_cnt)
      nsyn_avg_out2.append(nsyn)
      sem_avg_out2.append(sem_cnt)
      nsem_avg_out2.append(nsem)
      na_avg_out2.append(na_cnt)
      nna_avg_out2.append(nna)
    print(filename)
    print(lex_cnt, syn_cnt, sem_cnt, na_cnt)
    print(nlex, nsyn, nsem, nna)
    print('lex %.4f' % (lex_cnt / float(total_cnt)))
    print('syn %.4f' % (syn_cnt / float(total_cnt)))
    print('sem %.4f' % (sem_cnt / float(total_cnt)))
    print('na %.4f' % (na_cnt / float(total_cnt)))

print('Average, after contextualization')
total_cnt_1 = np.average(lex_avg_out1) + np.average(syn_avg_out1)\
  + np.average(sem_avg_out1) + np.average(na_avg_out1) 
print(lex_avg_out1, syn_avg_out1, sem_avg_out1, na_avg_out1)
print(nlex_avg_out1, nsyn_avg_out1, nsem_avg_out1, nna_avg_out1)
print(np.average(nlex_avg_out1), np.average(nsyn_avg_out1), 
  np.average(nsem_avg_out1), np.average(nna_avg_out1))
print('lex %.4f' % (np.average(lex_avg_out1) / float(total_cnt_1)))
print('syn %.4f' % (np.average(syn_avg_out1) / float(total_cnt_1)))
print('sem %.4f' % (np.average(sem_avg_out1) / float(total_cnt_1)))
print('na %.4f' % (np.average(na_avg_out1) / float(total_cnt_1)))

print('Average, before contextualization')
total_cnt_2 = np.average(lex_avg_out2) + np.average(syn_avg_out2)\
  + np.average(sem_avg_out2) + np.average(na_avg_out2)
print(lex_avg_out2, syn_avg_out2, sem_avg_out2, na_avg_out2)
print(nlex_avg_out2, nsyn_avg_out2, nsem_avg_out2, nna_avg_out2)
print(np.average(nlex_avg_out2), np.average(nsyn_avg_out2), 
  np.average(nsem_avg_out2), np.average(nna_avg_out2))
print('lex %.4f' % (np.average(lex_avg_out2) / float(total_cnt_2)))
print('syn %.4f' % (np.average(syn_avg_out2) / float(total_cnt_2)))
print('sem %.4f' % (np.average(sem_avg_out2) / float(total_cnt_2)))
print('na %.4f' % (np.average(na_avg_out2) / float(total_cnt_2)))