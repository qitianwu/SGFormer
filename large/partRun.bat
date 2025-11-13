@echo off
echo Running Indian_pines...
python main.py --method sgformer --dataset Indian_pines --metric oa --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 ^
 --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0.0 --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act ^
 --trans_num_layers 1 --trans_dropout 0.5 --trans_weight_decay 0.0 --trans_use_residual --trans_use_weight --trans_use_bn ^
 --seed 123 --runs 1 --epochs 5000  --eval_step 10 --device 0 --data_dir D:\hsi_dataset\Indian_pines ^
