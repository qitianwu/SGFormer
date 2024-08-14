from models import *
from ours import *
from oursSOFT import *
from oursGAT import *
from oursNF import *
from nodeformer import *
from difformer import *
from graphormer import *
from graphtrans import *
from graphgps import *


def parse_method(method, args, c, d, device):
    if method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
    elif method == 'sgc':
        model = SGC(in_channels=d,
                    out_channels=c,
                      hops=args.hops).to(device)
    elif method == 'gat':
        model = GAT(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn, 
                    heads=args.gat_heads, 
                    out_heads=args.out_heads).to(device)
    elif method == 'gcnjk':
        model = GCNJK(in_channels=d,
                      hidden_channels=args.hidden_channels,
                      out_channels=c,
                      num_layers=args.num_layers,
                      dropout=args.dropout).to(device)
    elif method == 'appnp':
        model = APPNP_Net(in_channels=d,
                          hidden_channels=args.hidden_channels,
                          out_channels=c,
                          dropout=args.dropout).to(device)
    elif method == 'h2gcn':
        model = model = H2GCN(feat_dim=d,hidden_dim=args.hidden_channels,class_dim=c,dropout=args.dropout).to(device)
    elif method == 'sign':
        model = SIGN(in_channels=d,
                     hidden_channels=args.hidden_channels,
                     out_channels=c,
                     hops=args.hops,
                     num_layers=args.num_layers,
                     dropout=args.dropout,
                     use_bn=args.use_bn).to(device)
    elif method == 'graphgps':
        model = GPSModel(in_channels=d,
                         out_channels=c,
                         hidden_channels=args.hidden_channels,
                         num_layers=args.num_layers,
                         num_heads=args.num_heads,
                         dropout=args.dropout,
                         attn_dropout=args.dropout,
                         use_bn=args.use_bn).to(device)
    elif method == 'graphormer':
        model = Graphormer(num_classes=c, 
                            encoder_embed_dim=args.encoder_emdim,
                            encoder_attention_heads=1, 
                            num_encoder_layers=2, 
                            dropout=args.dropout,
                            activation_dropout=args.dropout,
                            attention_dropout=args.dropout,
                            device=device).to(device)
    elif method == 'graphtrans':
        model = graphTrans(in_channels=d,
                         hidden_channels=args.hidden_channels,
                         out_channels=c,
                         gnn_emb_dim=64,
                         d_model=64,
                         num_layers=args.num_layers, 
                         num_trans_layers=2,
                         num_trans_head=1,
                         dim_feedforward=256,
                         dropout=args.dropout).to(device)
    elif method == 'nodeformer':
        model = NodeFormer(in_channels=d,
                         hidden_channels=args.hidden_channels,
                         out_channels=c,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         num_heads=args.num_heads,
                         use_bn=args.use_bn).to(device)
    elif method == 'difformer':
        model = DIFFormer(in_channels=d,
                          hidden_channels=args.hidden_channels,
                          out_channels=c,
                          num_layers=args.num_layers,
                          alpha=args.alpha,
                          dropout=args.dropout,
                          num_heads=args.num_heads).to(device)
    elif method == 'ours':
        if args.use_graph:
            if args.attention == 'gcn':
                gnn=parse_method(args.backbone, args, args.hidden_channels, d, device)
                model = SGFormer(d, args.hidden_channels, c, num_layers=args.ours_layers, alpha=args.alpha, dropout=args.ours_dropout, num_heads=args.num_heads,
                        use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate).to(device)
            elif args.attention == 'softmax':
                gnn=parse_method(args.backbone, args, args.hidden_channels, d, device)
                model = SGFormerSOFT(d, args.hidden_channels, c, num_layers=args.ours_layers, alpha=args.alpha, dropout=args.ours_dropout, num_heads=args.num_heads,
                     use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate).to(device)
            elif args.attention == 'gat':
                gnn=parse_method(args.backbone, args, args.hidden_channels, d, device)
                model = SGFormerGAT(d, args.hidden_channels, c, num_layers=args.ours_layers, alpha=args.alpha, dropout=args.ours_dropout, num_heads=args.num_heads,
                     use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate).to(device)
            elif args.attention == 'nodeformer':
                gnn=parse_method(args.backbone, args, args.hidden_channels, d, device)
                model = SGFormerNF(d, args.hidden_channels, c, num_layers=args.ours_layers, alpha=args.alpha, dropout=args.ours_dropout, num_heads=args.num_heads,
                     use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, gnn=gnn, aggregate=args.aggregate).to(device)

        else:
            model = Ours(d, args.hidden_channels, c, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout, num_heads=args.num_heads,
                     use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph, use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight, aggregate=args.aggregate).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    # parser.add_argument('--data_dir', type=str, default='../../../NodeFormer/data/')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
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
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=500,
                        help='Total number of test')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')

    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true',
                        help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--attention', type=str, default='gcn')
    
    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')

    parser.add_argument('--no_feat_norm', action='store_true',
                        help='Not use feature normalization.')

    # ours
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience.')
    parser.add_argument('--graph_weight', type=float,
                        default=0.8, help='graph weight.')
    parser.add_argument('--ours_weight_decay', type=float,
                         help='Ours\' weight decay. Default to weight_decay.')
    parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--backbone', type=str, default='gcn',
                        help='Backbone.')
    parser.add_argument('--ours_layers', type=int, default=2,
                        help='gnn layer.')
    parser.add_argument('--ours_dropout', type=float, 
                        help='gnn dropout.')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')
    
    # hyper-parameter for gnn baseline
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')
    
    # graphormer
    parser.add_argument('--num_elayers', type=int, default=2,
                        help='number of encoder layers for graphormers')
    parser.add_argument('--encoder_emdim', type=int, default=768,
                        help='number of encoder embedded dimension')


def parser_add_default_args(args):
    if args.method=='ours':
        if args.ours_weight_decay is None:
            args.ours_weight_decay=args.weight_decay
        if args.ours_dropout is None:
            args.ours_dropout=args.dropout