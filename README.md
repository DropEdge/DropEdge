 DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
====
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dropedge-towards-deep-graph-convolutional/node-classification-on-cora-full-supervised)](https://paperswithcode.com/sota/node-classification-on-cora-full-supervised?p=dropedge-towards-deep-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dropedge-towards-deep-graph-convolutional/node-classification-on-citeseer-full)](https://paperswithcode.com/sota/node-classification-on-citeseer-full?p=dropedge-towards-deep-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dropedge-towards-deep-graph-convolutional/node-classification-on-pubmed-full-supervised)](https://paperswithcode.com/sota/node-classification-on-pubmed-full-supervised?p=dropedge-towards-deep-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dropedge-towards-deep-graph-convolutional/node-classification-on-reddit)](https://paperswithcode.com/sota/node-classification-on-reddit?p=dropedge-towards-deep-graph-convolutional)

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
sh scripts/supervised/cora_IncepGCN.sh
```

## Data
The data format is same as [GCN](https://github.com/tkipf/gcn). We provide three benchmark datasets as examples (see `data` folder). We use the public dataset splits provided by [Planetoid](https://github.com/kimiyoung/planetoid). The semi-supervised setting strictly follows [GCN](https://github.com/tkipf/gcn), while the full-supervised setting strictly follows [FastGCN](https://github.com/matenure/FastGCN) and [ASGCN](https://github.com/huangwb/AS-GCN). 


## Benchmark Results
For the details of backbones in Tables, please refer to the Appendix B.2 in the paper. All results are obtained on GPU (CUDA Version 9.0.176). 
### Full-supervised Setting Results

The following table demonstrates the testing accuracy (%) comparisons on different backbones and layers w and w/o DropEdge.
<escape>
<table><tr><th rowspan="2">Dataset</th><th rowspan="2">Backbone</th><th colspan="2">2 layers</th><th colspan="2">4 layers</th><th colspan="2">8 layers</th><th colspan="2">16 layers</th><th colspan="2">32 layers</th><th colspan="2">64 layers</th></tr><tr><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td></tr><tr><td rowspan="5">Cora</td><td>GCN</td><td>86.10</td><td>86.50</td><td>85.50</td><td>87.60</td><td>78.70</td><td>85.80</td><td>82.10</td><td>84.30</td><td>71.60</td><td>74.60</td><td>52.00</td><td>53.20</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>86.00</td><td>87.00</td><td>85.40</td><td>86.90</td><td>85.30</td><td>86.90</td><td>85.10</td><td>86.80</td><td>79.80</td><td>84.80</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>86.90</td><td>87.70</td><td>86.70</td><td>87.80</td><td>86.20</td><td>88.00</td><td>87.10</td><td>87.60</td><td>86.30</td><td>87.90</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>85.60</td><td>87.90</td><td>86.70</td><td>88.20</td><td>87.10</td><td>87.70</td><td>87.40</td><td>87.70</td><td>85.30</td><td>88.20</td></tr><tr><td>GraphSage</td><td>87.80</td><td>88.10</td><td>87.10</td><td>88.10</td><td>84.30</td><td>87.10</td><td>84.10</td><td>84.50</td><td>31.90</td><td>32.20</td><td>31.90</td><td>31.90</td></tr><tr><td rowspan="5">Citeseer</td><td>GCN</td><td>75.90</td><td>78.70</td><td>76.70</td><td>79.20</td><td>74.60</td><td>77.20</td><td>65.20</td><td>76.80</td><td>59.20</td><td>61.40</td><td>44.60</td><td>45.60</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>78.90</td><td>78.80</td><td>77.80</td><td>78.80</td><td>78.20</td><td>79.40</td><td>74.40</td><td>77.90</td><td>21.20</td><td>75.30</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>79.10</td><td>80.20</td><td>79.20</td><td>80.20</td><td>78.80</td><td>80.10</td><td>71.70</td><td>80.00</td><td>76.70</td><td>80.00</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>79.50</td><td>79.90</td><td>79.60</td><td>80.50</td><td>78.50</td><td>80.20</td><td>72.60</td><td>80.30</td><td>79.00</td><td>79.90</td></tr><tr><td>GraphSage</td><td>78.40</td><td>80.00</td><td>77.30</td><td>79.20</td><td>74.10</td><td>77.10</td><td>72.90</td><td>74.50</td><td>37.00</td><td>53.60</td><td>16.90</td><td>25.10</td></tr><tr><td rowspan="5">Pubmed</td><td>GCN</td><td>90.20</td><td>91.20</td><td>88.70</td><td>91.30</td><td>90.10</td><td>90.90</td><td>88.10</td><td>90.30</td><td>84.60</td><td>86.20</td><td>79.70</td><td>79.00</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>90.70</td><td>90.70</td><td>89.60</td><td>90.50</td><td>89.60</td><td>91.00</td><td>90.20</td><td>91.10</td><td>87.90</td><td>90.20</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>90.50</td><td>91.30</td><td>90.60</td><td>91.20</td><td>89.90</td><td>91.50</td><td>89.20</td><td>91.30</td><td>90.60</td><td>91.60</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>89.90</td><td>91.60</td><td>90.20</td><td>91.50</td><td>90.80</td><td>91.30</td><td>OOM</td><td>90.50</td><td>OOM</td><td>90.00</td></tr><tr><td>GraphSage</td><td>90.10</td><td>90.70</td><td>89.40</td><td>91.20</td><td>90.20</td><td>91.70</td><td>83.50</td><td>87.80</td><td>41.30</td><td>47.90</td><td>40.70</td><td>62.30</td></tr></table>
</escape>

### Semi-supervised Setting Results
The following table demonstrates the testing accuracy (%) comparisons on different backbones and layers w and w/o DropEdge.
<table><tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><th colspan="2">2 layers</th><th colspan="2">4 laysers</th><th colspan="2">8 layers</th><th colspan="2">16 layers</th><th colspan="2">32 layers</th><th colspan="2">64 layers</th></tr><tr><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td></tr><tr><td rowspan="4">Cora</td><td>GCN</td><td>81.10</td><td>82.80</td><td>80.40</td><td>82.00</td><td>69.50</td><td>75.80</td><td>64.90</td><td>75.70</td><td>60.30</td><td>62.50</td><td>28.70</td><td>49.50</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>78.80</td><td>83.30</td><td>75.60</td><td>82.80</td><td>72.20</td><td>82.70</td><td>76.60</td><td>81.10</td><td>61.10</td><td>78.90</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>80.20</td><td>83.30</td><td>80.70</td><td>82.60</td><td>80.20</td><td>83.00</td><td>81.10</td><td>82.50</td><td>71.50</td><td>83.20</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>77.60</td><td>82.90</td><td>76.50</td><td>82.50</td><td>81.70</td><td>83.10</td><td>81.70</td><td>83.10</td><td>80.00</td><td>83.50</td></tr><tr><td rowspan="4">Citeseer</td><td>GCN</td><td>70.80</td><td>72.30</td><td>67.60</td><td>70.60</td><td>30.20</td><td>61.40</td><td>18.30</td><td>57.20</td><td>25.00</td><td>41.60</td><td>20.00</td><td>34.40</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>70.50</td><td>72.20</td><td>65.00</td><td>71.60</td><td>66.50</td><td>70.10</td><td>62.60</td><td>70.00</td><td>22.10</td><td>65.10</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>68.70</td><td>72.60</td><td>67.70</td><td>71.80</td><td>69.80</td><td>72.60</td><td>68.20</td><td>70.80</td><td>63.40</td><td>72.20</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>69.30</td><td>72.70</td><td>68.40</td><td>71.40</td><td>70.20</td><td>72.50</td><td>68.00</td><td>72.60</td><td>67.50</td><td>71.00</td></tr><tr><td rowspan="4">Pubmed</td><td>GCN</td><td>79.00</td><td>79.60</td><td>76.50</td><td>79.40</td><td>61.20</td><td>78.10</td><td>40.90</td><td>78.50</td><td>22.40</td><td>77.00</td><td>35.30</td><td>61.50</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>78.60</td><td>78.80</td><td>78.10</td><td>78.90</td><td>75.50</td><td>78.00</td><td>67.90</td><td>78.20</td><td>66.90</td><td>76.90</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>78.00</td><td>78.70</td><td>78.10</td><td>78.70</td><td>72.60</td><td>79.10</td><td>72.40</td><td>79.20</td><td>74.50</td><td>78.90</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>77.70</td><td>79.50</td><td>77.90</td><td>78.60</td><td>74.90</td><td>79.00</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

## Change Log
 * 2020-03-04: Support for `tensorboard` and added an example in `src/train_new.py`. Thanks for [MihailSalnikov](https://github.com/MihailSalnikov).
 * 2019-10-11: Support both full-supervised and semi-supervised task setting for `Cora`, `Citeseer` and `Pubmed`. See `--task_type` option.

## References
```
@inproceedings{
rong2020dropedge,
title={DropEdge: Towards Deep Graph Convolutional Networks on Node Classification},
author={Yu Rong and Wenbing Huang and Tingyang Xu and Junzhou Huang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Hkx1qkrKPr}
}
```



