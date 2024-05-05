# SGFormer: Simplified Graph Transformers

The official implementation for NeurIPS23 paper "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations".

Related material: [[Paper](https://arxiv.org/pdf/2306.10759.pdf)], [[Blog](https://zhuanlan.zhihu.com/p/674548352)], [[Video](https://www.bilibili.com/video/BV1Gx4y1r7ZQ/?spm_id_from=333.999.0.0&vd_source=dd4795a9e34dbf19550fff1087216477)]

SGFormer is a graph encoder backbone that efficiently computes all-pair interactions with one-layer attentive propagation.

SGFormer is built upon our previous works on scalable graph Transformers with linear complexity [NodeFormer](https://github.com/qitianwu/NodeFormer) (NeurIPS22, spotlight) and [DIFFormer](https://github.com/qitianwu/DIFFormer) (ICLR23, spotlight). 

## What's news

[2023.10.28] We release the code for the model on large graph benchmarks. More detailed info will be updated soon.

[2023.12.20] We supplement more details for how to run the code.

[2024.05.05] We supplement the code for testing time and memory in `./medium/time_test.py`

## Model and Results

The model adopts a simple architecture and is comprised of a one-layer global attention and a shallow GNN.

<img width="862" alt="image" src="https://github.com/qitianwu/SGFormer/assets/22075007/3b46ec4c-1532-4d8e-b2d7-3bc1368c7ef8">

The following tables present the results for standard node classification tasks on medium-sized and large-sized graphs.

<img width="1104" alt="image" src="https://github.com/qitianwu/SGFormer/assets/22075007/a017202b-14c5-490f-9ca5-dd21c755add4">

<img width="1099" alt="image" src="https://github.com/qitianwu/SGFormer/assets/22075007/aeb29fb7-8ac2-407d-9599-23f681b34672">

## Requirements

For datasets except ogbn-papers100M, we used the environment with package versions indicated in `./large/requirement.txt`. For ogbn-papers100M, one needs PyG version >=2.0 for running the code. 

## Dataset

One can download the datasets (Planetoid, Deezer, Pokec, Actor/Film) from the google drive link below:

https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link

For Chameleon and Squirrel, we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main) that filter out the overlapped nodes.

For the OGB datasets, they will be downloaded automatically when running the code.

## Run the codes

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline.

### Citation

If you find our code and model useful, please cite our work. Thank you!

```bibtex
      @inproceedings{
        wu2023sgformer,
        title={SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations},
        author={Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan},
        booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
        year={2023}
        }
```

