"""
Training on large dataset using neighbor sampling.
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import load_fixed_splits
from dataset import load_dataset
from parse import parse_method, parser_add_main_args
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils import add_self_loops, to_undirected


def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def train(model, graph, loss_func, optimizer, batch_size):
    model.train()
    output = model(graph.x, graph.edge_index)[:batch_size]
    labels = graph.y[:batch_size]
    loss = loss_func(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, batch_size):
    model = model.eval()
    output = model(graph.x, graph.edge_index)[:batch_size]
    labels = graph.y[:batch_size]
    correct = (output.argmax(-1) == labels).sum().item()
    total = labels.size(0)
    return correct, total


def make_print(args):
    method = args.method
    print_str = f"batch size: {args.batch_size} "
    if args.use_pretrained:
        print_str += f"use pretrained: {args.model_dir} "
    if method == "gcn":
        print_str += f"method: {args.method} layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay}\n"
    elif method == "ours":
        print_str += (
            f"method: {args.method} hidden:{args.hidden_channels} num_layers:{args.num_layers} ours_layers:{args.ours_layers} lr:{args.lr} alpha:{args.alpha} decay:{args.weight_decay}\n"
            + f"  dropout:{args.dropout} ours_dropout:{args.ours_dropout} epochs:{args.epochs} use_bn:{args.use_bn} use_residual:{args.use_residual}, use_weight:{args.use_weight}, use_init:{args.use_init}, use_act:{args.use_act}, ours_use_weight:{args.ours_use_weight},\n"
            + f"  ours_use_residual:{args.ours_use_residual}, ours_use_act:{args.ours_use_act} graph_weight:{args.graph_weight}\n"
        )

    print_str += (
        f" hidden:{args.hidden_channels} num_layers:{args.num_layers} ours_layers:1 lr:{args.lr} decay:{args.weight_decay}"
        + f" dropout:{args.dropout} ours_dropout:{args.ours_dropout} epochs:{args.epochs} graph_weight:{args.graph_weight}\n"
    )

    return print_str


def main():
    parser = argparse.ArgumentParser(description="General Training Pipeline")
    parser_add_main_args(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    # --- load data --- #
    print("Start loading dataset")
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    edge_index = to_undirected(dataset.graph["edge_index"])
    edge_index, _ = add_self_loops(edge_index, num_nodes=dataset.graph["num_nodes"])
    data = Data(x=dataset.graph["node_feat"], edge_index=edge_index, y=dataset.label)

    # get the splits, only support 1 split.
    if args.rand_split:
        split_idx_lst = [
            dataset.get_idx_split(
                train_prop=args.train_prop, valid_prop=args.valid_prop
            )
            for _ in range(1)
        ]
    elif args.rand_split_class:
        split_idx_lst = [
            dataset.get_idx_split(
                split_type="class", label_num_per_class=args.label_num_per_class
            )
            for _ in range(1)
        ]
    elif args.dataset in ["ogbn-papers100M"]:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(1)]
    else:
        split_idx_lst = load_fixed_splits(
            args.data_dir, dataset, name=args.dataset, protocol=args.protocol
        )

    split_idx = split_idx_lst[0]

    input_channels = data.x.shape[1]
    output_channels = data.y.max().item() + 1
    if len(data.y.shape) > 1:
        output_channels = max(output_channels, data.y.shape[1])

    data.y[data.y.isnan()] = 404.0
    data.y = data.y.view(-1)
    data.y = data.y.to(torch.long)

    device = torch.device(f"cuda:{args.device}")
    print("Finish loading dataset")

    # --- init model --- #
    model = parse_method(args, output_channels, input_channels, device)

    loss_func = nn.CrossEntropyLoss()

    print("Start sampling")
    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx["train"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )
    valid_loader = NeighborLoader(
        data,
        input_nodes=split_idx["valid"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx["test"],
        num_neighbors=[15, 10, 5],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
    )
    print("Finish sampling")

    results = []
    best_model = None
    for run in range(args.runs):
        best_val_acc, best_test_acc, best_epoch, highest_test_acc = 0, 0, 0, 0
        if args.use_pretrained:
            print(f"Load pretrained model from {args.model_dir}.")
            model.load_state_dict(torch.load(args.model_dir, map_location=device))
        else:
            model.reset_parameters()

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        for e in range(args.epochs):
            # --- train --- #
            tot_loss = 0
            for graph in train_loader:
                graph = graph.to(device)
                loss = train(model, graph, loss_func, optimizer, graph.batch_size)
                tot_loss += loss
            # --- valid ---#
            valid_correct, valid_tot = 0, 0
            for graph in valid_loader:
                graph = graph.to(device)
                correct, tot = evaluate(model, graph, graph.batch_size)
                valid_correct += correct
                valid_tot += tot
            val_acc = valid_correct / valid_tot
            # --- test --- #
            test_correct, test_tot = 0, 0
            for graph in test_loader:
                graph = graph.to(device)
                correct, tot = evaluate(model, graph, graph.batch_size)
                test_correct += correct
                test_tot += tot
            test_acc = test_correct / test_tot
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = e + 1
                if args.save_model:
                    best_model = copy.deepcopy(model)
            if test_acc > highest_test_acc:
                highest_test_acc = test_acc
            if args.display_step > 0 and (e == 0 or (e + 1) % args.display_step == 0):
                print(
                    f"Epoch: {e+1:02d} "
                    f"Loss: {tot_loss:.4f} "
                    f"Valid acc: {val_acc * 100:.2f}% "
                    f"Test acc: {test_acc * 100:.2f}%"
                )

        print(f"Run {run+1:02d}")
        print(f"Best epoch: {best_epoch}")
        print(f"Highest test acc: {highest_test_acc * 100:.2f}%")
        print(f"Valid acc: {best_val_acc * 100:.2f}%")
        print(f"Test acc: {best_test_acc * 100:.2f}%")

        results.append([highest_test_acc, best_val_acc, best_test_acc])

    results = torch.as_tensor(results) * 100  # (runs, 3)
    print_str = f"{results.shape[0]} runs: "
    r = results[:, 0]
    print_str += f"Highest Test: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 1]
    print_str += f"Best Valid: {r.mean():.2f} ± {r.std():.2f} "
    r = results[:, 2]
    print_str += f"Final Test: {r.mean():.2f} ± {r.std():.2f} "
    print_str += f"Best epoch: {best_epoch}"
    print(print_str)

    out_folder = "results"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    file_name = f"{args.dataset}_{args.method}.txt"
    out_path = os.path.join(out_folder, file_name)
    with open(out_path, "a+") as f:
        config_str = make_print(args)
        f.write(config_str)
        f.write(print_str)
        f.write("\n\n")

    if args.save_model:
        save_folder = "models"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        path = os.path.join(
            save_folder, f"{args.dataset}_{args.method}_{args.epochs}.pt"
        )
        torch.save(best_model.state_dict(), path)


if __name__ == "__main__":
    main()
