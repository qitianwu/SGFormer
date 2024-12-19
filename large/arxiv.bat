@echo off
echo Running ogbn-arxiv...
python main.py --method sgformer --dataset ogbn-arxiv --metric acc --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 ^
--gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0.0 --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act ^
--trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0.0 --trans_use_residual --trans_use_weight --trans_use_bn ^
--seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 0 --data_dir D:\ogb_dataset\ogb\arxiv
pause