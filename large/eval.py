import torch
import torch.nn.functional as F

from torch_geometric.utils import subgraph

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(torch.device(device))
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

def evaluate_batch(model, dataset, split_idx, args, device, n, true_label):
    num_batch = n // args.batch_size + 1
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    train_total, train_correct=0, 0
    valid_total, valid_correct=0, 0
    test_total, test_correct=0, 0

    with torch.no_grad():
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            train_mask_i = train_mask[idx_i]
            valid_mask_i = valid_mask[idx_i]
            test_mask_i = test_mask[idx_i]

            out_i = model(x_i, edge_index_i)

            cur_train_total, cur_train_correct=eval_acc(y_i[train_mask_i], out_i[train_mask_i])
            train_total+=cur_train_total
            train_correct+=cur_train_correct
            cur_valid_total, cur_valid_correct=eval_acc(y_i[valid_mask_i], out_i[valid_mask_i])
            valid_total+=cur_valid_total
            valid_correct+=cur_valid_correct
            cur_test_total, cur_test_correct=eval_acc(y_i[test_mask_i], out_i[test_mask_i])
            test_total+=cur_test_total
            test_correct+=cur_test_correct

            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            # valid_acc = eval_func(
            #     dataset.label[split_idx['valid']], out[split_idx['valid']])
            # test_acc = eval_func(
            #     dataset.label[split_idx['test']], out[split_idx['test']])
        train_acc=train_correct/train_total
        valid_acc=valid_correct/valid_total
        test_acc=test_correct/test_total

    return train_acc, valid_acc, test_acc, 0, None

def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred=torch.max(pred,dim=1,keepdim=True)[1]
    # cmp=torch.eq(true, pred)
    # print(f'pred:{pred}')
    # print(cmp)
    true_cnt=(true==pred).sum()

    return true.shape[0], true_cnt.item()


if __name__=='__main__':
    x=torch.arange(4).unsqueeze(1)
    y=torch.Tensor([[3,0,0,0],
                    [3,2,1.5,2.8],
                    [0,0,2,1],
                    [0,0,1,3]
                    ])
    a, b=eval_acc(x, y)
    print(x)
    print(a,b)
