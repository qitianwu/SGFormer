# best cora
python main.py --attention soft--backbone gcn --dataset cora --lr 0.01 --num_layers 2 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.5 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split --train_prop 0.05 --valid_prop 0.25 \
    --seed 123 --device 1 --epochs 1000 --runs 10 --data_dir ../data/ --patience 500

# best film
python main.py --attention soft --backbone gcn --dataset film --lr 0.001 --num_layers 2 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.2 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split --train_prop 0.05 --valid_prop 0.25 \
    --seed 123 --device 1 --epochs 1000 --runs 10 --data_dir ../data/ --patience 500