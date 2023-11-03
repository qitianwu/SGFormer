from ours import *

def parse_method(args, c, d, device):
    if args.method == 'ours':
        model = SGFormer(d, args.hidden_channels, c, trans_num_layers=args.ours_layers, alpha=args.alpha,
                     trans_dropout=args.ours_dropout, trans_num_heads=args.num_heads, trans_use_bn=args.use_bn, trans_use_residual=args.ours_use_residual, trans_use_weight=args.ours_use_weight, trans_use_act=args.ours_use_act,
                     gnn_num_layers=args.num_layers, gnn_dropout=args.dropout, gnn_use_bn=args.use_bn, gnn_use_residual=args.use_residual, gnn_use_weight=args.use_weight, gnn_use_init=args.use_init, gnn_use_act=args.use_act,
                    use_graph=args.use_graph, graph_weight=args.graph_weight, aggregate=args.aggregate).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')

    # model
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')

    # ours
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience.')
    parser.add_argument('--graph_weight', type=float,
                        default=0.8, help='graph weight.')
    parser.add_argument('--ours_weight_decay', type=float,
                        help='Ours\' weight decay. Default to weight_decay.')
    parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--ours_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--ours_layers', type=int, default=2,
                        help='gnn layer.')
    parser.add_argument('--ours_dropout', type=float,
                        help='gnn dropout.')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')


    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--model_dir', type=str, default='../../model/')


