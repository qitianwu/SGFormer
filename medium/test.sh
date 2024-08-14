#cora
python main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 2 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.9 \
    --method graphormer --ours_layers 1 --use_graph --graph_weight 1 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
   --rand_split --train_prop 0.01 --valid_prop 0.25 --no_feat_norm \
    --seed 123 --device 1 --runs 5 --data_dir ../data/


#citeseer
python main.py --backbone gcn --dataset citeseer --lr 0.01 --num_layers 2 \
    --hidden_channels 32 --weight_decay 0.0005 --dropout 0.5 \
    --method ours --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split --train_prop 0.02 --valid_prop 0.25 --no_feat_norm \
    --seed 123 --device 0 --runs 5 --data_dir ../data/

# pubmed
python main.py --backbone gcn --dataset pubmed --lr 0.01 --num_layers 2 \
    --hidden_channels 64 --weight_decay 0.0005 --dropout 0.5 \
     --method ours --ours_layers 1 --use_graph --graph_weight 0.5 \
    --ours_dropout 0.3 --use_residual --alpha 0.6 --ours_weight_decay 0.00005  \
    --rand_split --train_prop 0.3 --valid_prop 0.25 --no_feat_norm \
    --seed 123 --device 0 --runs 5 --data_dir ../data/


python main.py --backbone gcn --dataset pubmed --lr 0.01 --num_layers 2 \
    --hidden_channels 64 --weight_decay 0.0005 --dropout 0.5 \
     --method glognn --ours_layers 1 --use_graph --graph_weight 0.5 \
    --ours_dropout 0.3 --use_residual --alpha 0.6 --ours_weight_decay 0.00005  \
    --rand_split --train_prop 0.3 --valid_prop 0.25 --no_feat_norm \
    --seed 123 --device 0 --runs 5 --data_dir ../data/

python main.py --backbone gcn --dataset pubmed --lr 0.001 --num_layers 2 \
    --hidden_channels 128 --weight_decay 5e-3 --dropout 0.5 \
    --method gcn --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split --train_prop 0.04 --valid_prop 0.25 --no_feat_norm \
    --seed 123 --device 1 --runs 5 --data_dir ../data/

python time_test.py --backbone gcn --dataset cora --lr 0.01 --num_layers 2 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 2 --runs 1 --data_dir ../data/

python time_test.py --backbone gcn --dataset cora --lr 0.01 --num_layers 2 \
    --hidden_channels 32 --weight_decay 5e-4 --dropout 0.5 \
    --method difformer --ours_layers 2 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.2 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 1 --data_dir ../data/

python time_test.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 2 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 2 --runs 1 --data_dir ../data/

python time_test1.py --backbone gcn  --dataset pubmed  --num_layers 2 \
    --hidden_channels 128 --dropout 0.2   \
    --method difformer --use_graph --ours_use_residual --num_heads 1 \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --alpha 0.5  --device 0 --runs 1 --data_dir ../data/

# New
python time_test.py --backbone gcn --dataset cora --lr 0.005 --num_layers 8 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 2 --runs 1 --data_dir ../data/

python time_test.py --backbone gcn --dataset cora --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 2 --runs 1 --data_dir ../data/

python time_test.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 8 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 2 --runs 1 --data_dir ../data/

python time_test.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 4 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --method graphgps --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.3 --use_residual --alpha 0.5 --ours_weight_decay 0.01  \
    --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
    --seed 123 --device 3 --runs 1 --data_dir ../data/