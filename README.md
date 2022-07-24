code for:

Yao Fu, John Cunningham and Mirella Lapata, [Scaling Structured Inference with Randomization](https://arxiv.org/abs/2112.03638). ICML 2022

Yao Fu and Mirella Lapata, [Latent Topology Induction for Understanding Contextualized Representations](https://arxiv.org/abs/2206.01512). Arxiv 2022


# For reproducing _Scaling Structured Inference with Randomization_, table 2

Most experiemnts for table 2 are in:
```bash
cd src/rdp-exps
```

Experiments for estimating linear-chain CRF, partition function and entropy
* `src/rdp-exps/chaincrf.ipynb`

Experiments for estimating TreeCRF partition function
* `src/rdp-exps/TreeCRF.ipynb`

The influence of K1/K2 ratio
* `src/rdp-exps/TreeCRF_K1K2.ipynb`

Implementation of Linear-chain CRF RDP algorithms
* `src/frtorch/structure/linear_chain_crf.py`

Implementation of TreeCRF RDP algorithms
* 'src/rdp-exps/tree_crf.py'

# For reproducing _Latent Topology Induction for Understanding Contextualized Representations_ (coming soon)
