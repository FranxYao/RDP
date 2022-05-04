# Annotation Guidelines

This project aims to find what is encoded in a pretrained language model. You are given two files of word clusters. Each cluster is extracted from a pretrained language model. Your task is to annotate if words within a given cluster are
* LEX: lexically similar or related
* SYN: syntactically similar 
* SEM: semantically related to each other
* NA: none of the above
See below for more detailed explanations about the definition and examples of these classes.

Table of Content: 
- [Annotation Guidelines](#annotation-guidelines)
  - [File Format](#file-format)
  - [Examples of LEX cluster](#examples-of-lex-cluster)
  - [Examples of SYN cluster](#examples-of-syn-cluster)
  - [Examples of SEM cluster](#examples-of-sem-cluster)
  - [Differences between SYN and SEM](#differences-between-syn-and-sem)
  - [If the clusters can be of more than one classes](#if-the-clusters-can-be-of-more-than-one-classes)
  - [If you are still not sure](#if-you-are-still-not-sure)

## File Format
The two .txt (output1.txt and output2.txt) files provided to you are of the following format: 
```
state 257 cnt 1030 type=?
these 516 | them 359 | those 155 | 
```
where 
* "state 257" = the word cluster whose index is 257
* "cnt 1030" = the total occurrence of this state within the data is 1030, i.e., it appears 1030 times 
* "these 516" = the word "these" appears 516 times within this cluster (out of 1030 occurrences)
* The words are ranked according to their occurrence. Frequent words will come before infrequent words
* Similarly, clusters are ranked according to their occurrence. Frequent clusters will come before infrequent clusters

This is a LEX cluster, because all these words stem from the same word "they". So you just put a "LEX" at the end of the first line, like:
```
state 257 cnt 1030 type=LEX
these 516 | them 359 | those 155 | 
```

You should directly fill your answers in the provided files. There is not much cases, so generally you could finish labeling within an hour. You may label slowly at the beginning to gain a general feeling of the clusters. Once you feel like familiar enough, you can speed up. 

After you have finished, send the two files (output1.txt and output2.txt) back to yao.fu@ed.ac.uk

## Examples of LEX cluster
Other examples of LEX clusters are: 
```
state 837 cnt 2703 type=LEX
's 1489 | 'm 319 | 've 279 | 're 222 | 'd 203 | 'll 191
```
This is a LEX cluster because most tokens within this cluster are suffix

```
state 1366 cnt 973 type=LEX
90 25 | 80 24 | 9 23 | 256 17 | 93 16 | 130 12 | 128 11 | 286 11 | 91 11 | 129 10
```
This is a LEX cluster because most tokens are numbers 

```
state 1314 cnt 273 type=LEX
ie 32 | pp 26 | nt 26 | bnr 17 | gmt 12 | ch 10 | gm 8 | md 8 | rtsg 7 | chromium 7 
```
This is a LEX cluster because most tokens are acronyms

```
state 1788 cnt 267 type=LEX
clh 24 | cl 9 | vram 8 | cray 7 | clesun 7 | cr 5 | br 5
```
This is a LEX cluster because most of the tokens starts with a "c"

## Examples of SYN cluster

```
state 1245 cnt 828 SYN
me 350 | them 193 | person 74 | him 70 | people 57 | everyone 45
```
This is a SYN cluster because most tokens are pronouns

```
state 1402 cnt 578 SYN
between 167 | into 152 | under 125 | upon 40 | despite 22 | behind 22
```
This is a SYN cluster because most tokens are prepositions

```
state 1530 cnt 536 SYN
christians 32 | braves 14 | guns 14 | disks 11 | roads 9 | bruins 9 
```
This is a SYN cluster because most tokens are plural nouns

```
state 1329 cnt 385 SYN
based 43 | made 21 | produced 15 | replaced 15 | handled 8 | handed 7 | corrected 6 | printed 6 | measured 6 | updated 5
```
This is a SYN cluster because most tokens are verbs of their past time form

## Examples of SEM cluster
```
state 581 cnt 269 SEM
evidence 107 | suggest 41 | indicates 14 | indicate 14 | suggested 13 | imply 13 | implies 12
```
This is a SEM cluster because most of the words are about indicate / suggest/ imply evidences. 

```
state 1101 cnt 235 SEM
bible 108 | scripture 24 | mathew 10 | psalm 7 | prophet 7 | biblical 6
```
This is a SEM cluster because most of the words are about religions. 

## Differences between SYN and SEM
Syntactic clusters (SYN) are different than (SEM) as words within the syntactic cluster **do not have related meaning**. For example: 
```
state 1530 cnt 536 SYN
christians 32 | braves 14 | guns 14 | disks 11 | roads 9 | bruins 9 
```
This is a SYN cluster. "disks" is not related to "roads", and also not related to "christians". But they are all plural nouns. 

```
state 156 cnt 261 SEM
paper 39 | report 34 | reports 31 | published 23 | documents 21 | documentation 19
```
This is a SEM cluster because all these words about about publishing papers/ documents/ reports. They are of the same mearning group

## If the clusters can be of more than one classes
For example, if the cluster is like:
```
papers 101 | pens 82 | tables 37 | chairs 22 | computers 12
```
These are all words about office tools (SEM) and also plural nouns (SYN). In this case choose SEM over SYN

Another example is:
```
papers 101 | one 70 | pens 82 | tables 37 | ten 23 | chairs 22 | computers 12 | three 5 | four 4
```
This is a cluster of both office tools and numbers. In such situation, choose the majority class. Since the count of papers and pens and tables = 101 + 82 + 37, larger than the count of one and ten and three = 70 + 23 + 5. So this is a SEM cluster

Similarly, if a cluster contain words that looks liks noise, like:
```
known 43 | pub 34 | similar 34 | related 28 | various 23 | relevant 15 | derived 13 | compare 12 | compared 11 | compatible 10
```
where the words "similar", "related", "compare" are semantically close to each other, but the word "pub" and "known" seems to be far away. In this case, since "similar", "related", "compare" are of the majority, we label this cluster as a SEM cluster

## If you are still not sure 
If you cannot decide if a cluster is more like SYN or SEM, trust your instinct -- human instinct itself is a is value tool for evaluating AI models. 

If you cannot make out any sense from a cluster, set is as NA. For example: 
```
state 665 cnt 334 NA
regards 22 | morality 16 | nature 13 | respect 12 | sensitivity 10 | attitude 9 | behavior 8 | weakness 7 | orientation 6 | relativity 6
```
The words within this cluster seems not be close to each other. Even there exist related words like "regards" and "respect", they do not take a large portion of the cluster. So this cluster is labeled NA. 

