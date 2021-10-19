# T-SVDNet: Exploring High-Order Prototypical Correlations for Multi-Source Domain Adaptation (ICCV 2021, official Pytorch implementation)
![Teaser](docs/overview.png)
### The paper is available here: [Arxiv](https://arxiv.org/abs/2107.14447)
<!-- <br> -->
## Abstract
>Most existing domain adaptation methods focus on adaptation from only one source domain, however, in practice there are a number of relevant sources that could be leveraged to help improve performance on target domain. We propose a novel approach named T-SVDNet to address the task of Multi-source Domain Adaptation (MDA), which is featured by incorporating Tensor Singular Value Decomposition (T-SVD) into a neural network's training pipeline. Overall, high-order correlations among multiple domains and categories are fully explored so as to better bridge the domain gap. Specifically, we impose Tensor-Low-Rank (TLR) constraint on a tensor obtained by stacking up a group of prototypical similarity matrices, aiming at capturing consistent data structure across different domains. Furthermore, to avoid negative transfer brought by noisy source data, we propose a novel uncertainty-aware weighting strategy to adaptively assign weights to different source domains and samples based on the result of uncertainty estimation. Extensive experiments conducted on public benchmarks demonstrate the superiority of our model in addressing the task of MDA compared to state-of-the-art methods.

## Installation
```bash
pip install -r requirements.txt
```
## Data Preparation
Download [Digits-Five](https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm), [DomainNet](http://ai.bu.edu/M3SDA/) and [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017).
