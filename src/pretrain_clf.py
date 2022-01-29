import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from ogb.graphproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import Criterion, Writer, get_data_loaders, get_model, save_checkpoint, set_seed, process_data
from utils import get_preds, get_lr, get_local_config_name, write_stat_from_metric_dicts, init_metric_dict


def train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                       model=None, loaders=None, num_class=None, aux_info=None):
    if model is None:
        print('====================================')
        print('====================================')
        print(f'[INFO] Using device: {device}')
        print(f'[INFO] Using random_state: {random_state}')
        print(f'[INFO] Using dataset: {dataset_name}')
        print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    scheduler_config = model_config.get('pretrain_scheduler', {})
    assert model_config['model_name'] == model_name

    if model is None:
        batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state)

        model_config['deg'] = aux_info['deg']
        model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    else:
        print('[INFO] Using the given loaders and model architecture')

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    epochs, lr, wd = model_config['pretrain_epochs'], model_config['pretrain_lr'], model_config.get('pretrain_wd', 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    metric_dict = train(optimizer, scheduler, dataset_name, model, device, loaders, epochs, log_dir, writer, num_class, metric_dict, random_state,
                        model_config.get('use_edge_attr', True), aux_info['multi_label'])
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


@torch.no_grad()
def eval_one_batch(data, model, criterion, optimizer=None):
    assert optimizer is None
    model.eval()
    logits = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    loss = criterion(logits, data.y)
    return loss.item(), logits.data.cpu(), data.y.data.cpu()


def train_one_batch(data, model, criterion, optimizer):
    model.train()

    logits = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    loss = criterion(logits, data.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), logits.data.cpu(), data.y.data.cpu()


def run_one_epoch(dataset_name, data_loader, model, criterion, optimizer, epoch, phase, device, writer, random_state, use_edge_attr, multi_label):
    loader_len = len(data_loader)
    all_preds, all_targets, all_batch_losses, all_logits = [], [], [], []

    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        data = process_data(data, use_edge_attr)
        loss, logits, targets = run_one_batch(data.to(device), model, criterion, optimizer)
        preds = get_preds(logits, multi_label)

        acc = 0 if multi_label else (preds == targets).sum().item() / targets.shape[0]
        all_preds.append(preds), all_targets.append(targets), all_batch_losses.append(loss), all_logits.append(logits)
        desc = f'[Seed: {random_state}, Epoch: {epoch}]: {phase}........., loss: {loss:.3f}, acc: {acc:.3f}, '

        if idx == loader_len - 1:
            all_preds, all_targets, all_logits = np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_logits)
            all_acc = (all_preds == all_targets).sum() / (all_targets.shape[0] * all_targets.shape[1]) if multi_label else (all_preds == all_targets).sum().item() / all_targets.shape[0]

            desc = f'[Seed: {random_state}, Epoch: {epoch}]: {phase} finished, loss: {np.mean(all_batch_losses):.3f}, acc: {all_acc:.3f}, '

            writer.add_scalar(f'clf/{phase}/loss', np.mean(all_batch_losses), epoch)
            writer.add_scalar(f'clf/{phase}/acc', all_acc, epoch)
            auroc = None
            if 'ogb' in dataset_name:
                evaluator = Evaluator(name='-'.join(dataset_name.split('_')))
                auroc = evaluator.eval({'y_pred': all_logits, 'y_true': all_targets})['rocauc']
                desc += f'auroc: {auroc:.3f}'
                writer.add_scalar(f'clf/{phase}/auroc', auroc, epoch)

        pbar.set_description(desc)
    return auroc if auroc is not None else all_acc, np.mean(all_batch_losses)


def train(optimizer, scheduler, dataset_name, model, device, loaders, epochs, model_dir, writer, num_class, metric_dict, random_state,
          use_edge_attr, multi_label):
    criterion = Criterion(num_class, multi_label)

    for epoch in range(epochs):
        train_res, _ = run_one_epoch(dataset_name, loaders['train'], model, criterion, optimizer, epoch, 'train', device, writer, random_state, use_edge_attr, multi_label)
        valid_res, valid_loss = run_one_epoch(dataset_name, loaders['valid'], model, criterion, None, epoch, 'valid', device, writer, random_state, use_edge_attr, multi_label)
        test_res, _ = run_one_epoch(dataset_name, loaders['test'], model, criterion, None, epoch, 'test', device, writer, random_state, use_edge_attr, multi_label)

        writer.add_scalar('clf/lr', get_lr(optimizer), epoch)
        if scheduler is not None:
            scheduler.step(valid_res)
        if (valid_res > metric_dict['metric/best_clf_valid']) or (valid_res == metric_dict['metric/best_clf_valid'] and valid_loss < metric_dict['metric/best_clf_valid_loss']):
            metric_dict['metric/best_clf_epoch'] = epoch
            metric_dict['metric/best_clf_train'], metric_dict['metric/best_clf_valid'], metric_dict['metric/best_clf_test'] = train_res, valid_res, test_res
            metric_dict['metric/best_clf_valid_loss'] = valid_loss
            save_checkpoint(model, model_dir, model_name=f'epoch_{epoch}')

        for metric, value in metric_dict.items():
            metric = metric.split('/')[-1]
            writer.add_scalar(f'clf/{metric}', value, epoch)

        if epoch == epochs - 1:
            save_checkpoint(model, model_dir, model_name=f'epoch_{epoch}')

        print(f'[Seed: {random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
              f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
        print('====================================')
        print('====================================')
    return metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pretrain classifier')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-pretrain')
        hparam_dict, metric_dict = train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99' + '-pretrain' + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
