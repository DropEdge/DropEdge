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
sh scripts/supervised/cora_IncepGCN.sh
```

## Data
The data format is same as [GCN](https://github.com/tkipf/gcn). We provide three benchmark datasets as examples (see `data` folder). We use the public dataset splits provided by [Planetoid](https://github.com/kimiyoung/planetoid). The full-supervised setting strictly follows [GCN](https://github.com/tkipf/gcn), while the semi-supervised setting strictly follows [FastGCN](https://github.com/matenure/FastGCN) and [ASGCN](https://github.com/huangwb/AS-GCN). 


## Benchmark Results
For the details of backbones in Tables, please refer to the Appendix B.2 in papers. All results are obtained on GPU (CUDA Version 9.0.176). 
### Full-supervised Setting Results

The following table demonstrates the testing accuracy (%) comparisons on different backbones and layers w and w/o DropEdge.
<escape>
<table><tr><th rowspan="2">Dataset</th><th rowspan="2">Backbone</th><th colspan="2">2 layers</th><th colspan="2">4 layers</th><th colspan="2">8 layers</th><th colspan="2">16 layers</th><th colspan="2">32 layers</th><th colspan="2">64 layers</th></tr><tr><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td></tr><tr><td rowspan="5">Cora</td><td>GCN</td><td>86.10</td><td>86.50</td><td>85.50</td><td>87.60</td><td>78.70</td><td>85.80</td><td>82.10</td><td>84.30</td><td>71.60</td><td>74.60</td><td>52.00</td><td>53.20</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>86.00</td><td>87.00</td><td>85.40</td><td>86.90</td><td>85.30</td><td>86.90</td><td>85.10</td><td>86.80</td><td>79.80</td><td>84.80</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>86.90</td><td>87.70</td><td>86.70</td><td>87.80</td><td>86.20</td><td>88.00</td><td>87.10</td><td>87.60</td><td>86.30</td><td>87.90</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>85.60</td><td>87.90</td><td>86.70</td><td>88.20</td><td>87.10</td><td>87.70</td><td>87.40</td><td>87.70</td><td>85.30</td><td>88.20</td></tr><tr><td>GraphSage</td><td>87.80</td><td>88.10</td><td>87.10</td><td>88.10</td><td>84.30</td><td>87.10</td><td>84.10</td><td>84.50</td><td>31.90</td><td>32.20</td><td>31.90</td><td>31.90</td></tr><tr><td rowspan="5">Citeseer</td><td>GCN</td><td>75.90</td><td>78.70</td><td>76.70</td><td>79.20</td><td>74.60</td><td>77.20</td><td>65.20</td><td>76.80</td><td>59.20</td><td>61.40</td><td>44.60</td><td>45.60</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>78.90</td><td>78.80</td><td>77.80</td><td>78.80</td><td>78.20</td><td>79.40</td><td>74.40</td><td>77.90</td><td>21.20</td><td>75.30</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>79.10</td><td>80.20</td><td>79.20</td><td>80.20</td><td>78.80</td><td>80.10</td><td>71.70</td><td>80.00</td><td>76.70</td><td>80.00</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>79.50</td><td>79.90</td><td>79.60</td><td>80.50</td><td>78.50</td><td>80.20</td><td>72.60</td><td>80.30</td><td>79.00</td><td>79.90</td></tr><tr><td>GraphSage</td><td>78.40</td><td>80.00</td><td>77.30</td><td>79.20</td><td>74.10</td><td>77.10</td><td>72.90</td><td>74.50</td><td>37.00</td><td>53.60</td><td>16.90</td><td>25.10</td></tr><tr><td rowspan="5">Pubmed</td><td>GCN</td><td>90.20</td><td>91.20</td><td>88.70</td><td>91.30</td><td>90.10</td><td>90.90</td><td>88.10</td><td>90.30</td><td>84.60</td><td>86.20</td><td>79.70</td><td>79.00</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>90.70</td><td>90.70</td><td>89.60</td><td>90.50</td><td>89.60</td><td>91.00</td><td>90.20</td><td>91.10</td><td>87.90</td><td>90.20</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>90.50</td><td>91.30</td><td>90.60</td><td>91.20</td><td>89.90</td><td>91.50</td><td>89.20</td><td>91.30</td><td>90.60</td><td>91.60</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>89.90</td><td>91.60</td><td>90.20</td><td>91.50</td><td>90.80</td><td>91.30</td><td>OOM</td><td>90.50</td><td>OOM</td><td>90.00</td></tr><tr><td>GraphSage</td><td>90.10</td><td>90.70</td><td>89.40</td><td>91.20</td><td>90.20</td><td>91.70</td><td>83.50</td><td>87.80</td><td>41.30</td><td>47.90</td><td>40.70</td><td>62.30</td></tr></table>
</escape>

### Semi-supervised Setting Results
The following table demonstrates the testing accuracy (%) comparisons on different backbones and layers w and w/o DropEdge.
<table><tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><th colspan="2">2 layers</th><th colspan="2">4 layers</th><th colspan="2">8 layers</th><th colspan="2">16 layers</th><th colspan="2">32 layers</th><th colspan="2">64 layers</th></tr><tr><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td><td>Orignal</td><td>DropEdge</td></tr><tr><td rowspan="4">Cora</td><td>GCN</td><td>0.811</td><td>0.828</td><td>0.804</td><td>0.82</td><td>0.695</td><td>0.758</td><td>0.649</td><td>0.757</td><td>0.603</td><td>0.625</td><td>0.287</td><td>0.495</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>0.788</td><td>0.833</td><td>0.756</td><td>0.828</td><td>0.722</td><td>0.827</td><td>0.766</td><td>0.811</td><td>0.611</td><td>0.789</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>0.802</td><td>0.833</td><td>0.807</td><td>0.826</td><td>0.802</td><td>0.83</td><td>0.811</td><td>0.825</td><td>0.715</td><td>0.832</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>0.776</td><td>0.829</td><td>0.765</td><td>0.825</td><td>0.817</td><td>0.831</td><td>0.817</td><td>0.831</td><td>0.8</td><td>0.835</td></tr><tr><td rowspan="4">Citeseer</td><td>GCN</td><td>0.708</td><td>0.723</td><td>0.676</td><td>0.706</td><td>0.302</td><td>0.614</td><td>0.183</td><td>0.572</td><td>0.25</td><td>0.416</td><td>0.2</td><td>0.344</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>0.705</td><td>0.722</td><td>0.65</td><td>0.716</td><td>0.665</td><td>0.701</td><td>0.626</td><td>0.7</td><td>0.221</td><td>0.651</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>0.687</td><td>0.726</td><td>0.677</td><td>0.718</td><td>0.698</td><td>0.726</td><td>0.682</td><td>0.708</td><td>0.634</td><td>0.722</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>0.693</td><td>0.727</td><td>0.684</td><td>0.714</td><td>0.702</td><td>0.725</td><td>0.68</td><td>0.726</td><td>0.675</td><td>0.71</td></tr><tr><td rowspan="4">Pubmed</td><td>GCN</td><td>0.79</td><td>0.796</td><td>0.765</td><td>0.794</td><td>0.612</td><td>0.781</td><td>0.409</td><td>0.785</td><td>0.224</td><td>0.77</td><td>0.353</td><td>0.615</td></tr><tr><td>ResGCN</td><td>-</td><td>-</td><td>0.786</td><td>0.788</td><td>0.781</td><td>0.789</td><td>0.755</td><td>0.78</td><td>0.679</td><td>0.782</td><td>0.669</td><td>0.769</td></tr><tr><td>JKNet</td><td>-</td><td>-</td><td>0.78</td><td>0.787</td><td>0.781</td><td>0.787</td><td>0.726</td><td>0.791</td><td>0.724</td><td>0.792</td><td>0.745</td><td>0.789</td></tr><tr><td>IncepGCN</td><td>-</td><td>-</td><td>0.777</td><td>0.795</td><td>0.779</td><td>0.786</td><td>0.749</td><td>0.79</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

## Change Log
 * 2019-10-11: Support both full-supervised and semi-supervised task setting for `Cora`, `Citeseer` and `Pubmed`. See `--task_type` option.

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



