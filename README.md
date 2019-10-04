 DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
====
This is a Pytorch implementation of paper: DropEdge: Towards Deep Graph Convolutional Networks on Node Classification


## Requirements

  * Python 3.6.2
  * For the other packages, please refer to the requirements.txt.


## Usage
To run the demo:
```sh run.sh```

All scripts of different models with parameters for Cora, Citeseer and Pubmed are in `scripts` folder. You can reproduce the results by:
```
pip install -r requirements.txt
sh scripts/cora_IncepGCN.sh
```


## References
```
@inproceedings{
anonymous2020dropedge,
title={DropEdge: Towards Deep Graph Convolutional Networks on Node Classification},
author={Anonymous},
booktitle={Submitted to International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Hkx1qkrKPr},
note={under review}
}
```



