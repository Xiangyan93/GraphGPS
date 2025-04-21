# GraphGPS: General Powerful Scalable Graph Transformers
This repo is developed based on [GraphGPS](https://github.com/rampasek/GraphGPS). 

You can simply train and predict on your own datasets using this repo.

### Python environment setup with mamba

```bash
mamba create -n graphgps python=3.12
mamba activate graphgps

pip install git+https://gitlab.com/Xiangyan93/graphdot.git@feature/xy mgktools torch-scatter torch-sparse torch-geometric pytorch-lightning yacs torchmetrics ogb performer-pytorch git+https://github.com/Xiangyan93/GraphGPS.git@CustomDataset
```

### Running GraphGPS on your own datasets
```bash
mamba activate graphgps
graphgps_train -h
graphgps_predict -h
```
