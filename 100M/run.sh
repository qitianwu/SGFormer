
# pretrain
python nb-sample.py --dataset ogbn-papers100M --method ours --lr 0.001 --num_layers 3 \
    --hidden_channels 256 --dropout 0.2 --weight_decay 1e-5 --use_residual --use_weight --use_bn --use_init --use_act \
    --ours_layers 1 --ours_dropout 0.5 --ours_use_residual --ours_use_weight --ours_use_bn \
    --use_graph --graph_weight 0.8 \
    --batch_size 1000  --seed 123 --runs 1 --epochs 23 --display_step 5 --device 3 --save_model --data_dir /home/ubuntu

# finetune
python nb-sample.py --dataset ogbn-papers100M --data_dir /home/ubuntu --method ours --lr 0.0001 --num_layers 3 \
    --hidden_channels 256 --dropout 0.2 --weight_decay 1e-5 --use_residual --use_weight --use_bn --use_init --use_act \
    --ours_layers 1 --ours_dropout 0.5 --ours_use_residual --ours_use_weight --ours_use_bn \
    --use_graph --graph_weight 0.8 \
    --batch_size 1000  --seed 123 --runs 1 --epochs 10 --display_step 5 --device 3 --save_model \
    --use_pretrained --model_dir models/ogbn-papers100M_ours_23.pt
