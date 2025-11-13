import torch

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]  # run个空列表的列表

    def add_result(self, run, result):
        assert len(result) == 2  # check length 这里增加了 aa, kpp 6个参数。 从4->10
        # version2 = 这里只记录oa/loss/保存最优模型 最后评测即可
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)  # 对应列表增加结果

    def print_best_epoch(self, step=10):
        result = torch.tensor(self.results[0])
        ind = result[:, 1].argmin().item()
        print(f'Chosen epoch: {(ind+1)*step}')
        print(f"Final Valid Loss: {result[ind, 1]:.4f}")

    def print_statistics(self, run=None, mode='max_acc', step=10):
        # 不再代码改进阶段不再使用
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax  # 选择训练集最大精确度
            else:
                ind = argmin
            print_str = f'Run {run + 1:02d}:' + \
                f'Highest Train OA: {result[:, 0].max():.2f} ' + \
                f'Highest Valid OA: {result[:, 1].max():.2f} ' + \
                f'Highest Test OA: {result[:, 2].max():.2f}\n' + \
                f'Chosen epoch: {(ind+1)*step}\n' + \
                f'Final Train OA: {result[ind, 0]:.2f} ' + \
                f'Final Valid OA: {result[ind, 1]:.2f} ' + \
                f'Final Test OA: {result[ind, 2]:.2f}\n' + \
                f'Final Train AA: {result[ind, 4]:.2f} ' + \
                f'Final Valid AA: {result[ind, 5]:.2f} ' + \
                f'Final Test AA: {result[ind, 6]:.2f}\n' + \
                f'Final Train KPP: {result[ind, 7]:.2f} ' + \
                f'Final Valid KPP: {result[ind, 8]:.2f} ' + \
                f'Final Test KPP: {result[ind, 9]:.2f}\n'
            print(print_str)
            self.test = result[ind, 2]  # 存下final的test_AA
        else:
            best_results = []
            max_val_epoch=0
            for r in self.results:  # loop for runs
                r = 100*torch.tensor(r)
                # 取所有epoch中的最大
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                    max_val_epoch=r[:, 1].argmax()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            # best_result中有run组 数据
            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'Final Train OA: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'Final Test OA: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'resu lts/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
            f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
            f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")