""" ISBI 2022 """
import monai
import monai.transforms as mt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
import collections, json, argparse, time, pathlib
from typing import Callable, Dict, Optional, Tuple, Union


def get_q_hat(calibration_scores, labels, alpha=0.05):
    if not isinstance(calibration_scores, torch.Tensor):
        calibration_scores = torch.tensor(calibration_scores)
        
    n = calibration_scores.shape[0]
    
    #  sort scores and returns values and index that would sort classes
    values, indices = calibration_scores.sort(dim=1, descending=True)
    
    #  sum up all scores cummulatively and return to original index order 
    cum_scores = values.cumsum(1).gather(1, indices.argsort(1))[range(n), labels]
    
    #  get quantile with small correction for finite sample sizes
    q_hat = torch.quantile(cum_scores, np.ceil((n + 1) * (1 - alpha)) / n)
    
    return q_hat


def conformal_inference(scores, q_hat):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    assert q_hat < 1, 'q_hat must be below 1'
        
    n = scores.shape[0]
    
    values, indices = scores.sort(dim=1, descending=True)
    
    #  number of each confidence prediction set to acheive coverage
    set_sizes = (values.cumsum(1) > q_hat).int().argmax(dim=1)
    
    confidence_sets = [indices[i][0:(set_sizes[i] + 1)] for i in range(n)]
    
    return [x.tolist() for x in confidence_sets]


def average_models(models: dict):
    pass
#     for k in global_state.keys():
#         for l in local_state:
#             global_state[k] += l[k]
#         # +1 because for global model 
#         global_state[k] = torch.true_divide(global_state[k], len(local_state) + 1)
#     global_model.load_state_dict(global_state)
#     torch.save({
#         'state_dict': global_model.state_dict(),
#         'round': r,
#     }, str(rnd_dir / f'global_model_{r}.pth'))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.patients = df.patient.unique()
        self.classes = df.label.unique()
        self.transforms = mt.Compose([
            mt.LoadImaged('image'),
            mt.AsChannelFirstd('image'),
            mt.ToTensord('image'),
        ])
    
    def __getitem__(self, index: int) -> Tuple:
        patient = self.patients[index]
        exams = self.df.query(f'patient == {patient}')
        n = exams .shape[0]
        i, j = np.random.randint(0, high=n, size=2)
        fst_exam = self.transforms(exams.iloc[i])
        snd_exam = self.transforms(exams.iloc[j])
        fst_image = fst_exam['image']
        fst_label = fst_exam['label']
        fst_race = fst_exam['race']
        snd_image = snd_exam['image']
        snd_label = snd_exam['label']
        snd_race = snd_exam['race']
        return (
            fst_image,  # image array
            fst_label,  # target label
            fst_race,   # patient race
            snd_image, 
            snd_label,
            snd_race,
        )
        
    def __len__(self) -> int:
        return self.patients.shape[0]
    

def get_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = True,
        sampler=None,
    ) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    
def get_model(
        arch: str = 'DenseNet121',
        pretrained: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> torch.nn.Module:
    model = getattr(monai.networks.nets, arch)(
        pretrained=pretrained,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return model


def train_loop(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
        metric_functions: Optional[Callable] = None,
        epochs: int = 20,
        debug: bool = False,
    ) -> Union[Dict, Tuple[Dict, Dict]]:
    history = {}
    model = model.to('cuda')
    model.train()
    for i in range(1, epochs if not debug else 1 + 1):
        sum_loss = 0
        all_scores = []
        all_targets = []
        for j, batch in enumerate(loader):
            x1, y1, z1, *_ = batch
            x1 = x1.to('cuda')
            y1 = y1.to('cuda')
            scores = model(x1)
            loss = criterion(scores, y1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            targets = y1
            all_scores.append(scores.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            
        if debug: print('SCORES\n', all_scores[0])
        if debug: print('\nTARGETS\n', all_targets[0])
        
        sum_loss /= len(loader)
        print(f'\tEPOCH: {str(i).zfill(3)} {sum_loss:.4f}')
        history[i] = metric_functions(
            np.concatenate(all_scores), 
            np.concatenate(all_targets),
        )
        if valid_loader is None:
            continue
        
        model.eval()
        with torch.no_grad():
            valid_history = {}
            sum_loss = 0
            all_scores = []
            all_targets = []
            for j, batch in enumerate(loader):
                x1, y1, z1, *_ = batch
                x1 = x1.to('cuda')
                y1 = y1.to('cuda')
                scores = model(x1)
                loss = criterion(scores, y1)
                sum_loss += loss.item()
                targets = y1
                all_scores.append(scores.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

            sum_loss /= len(loader)
            print(f'\t\tVALID: {sum_loss:.4f}')
            valid_history[i] = metric_functions(
                np.concatenate(all_scores), 
                np.concatenate(all_targets),
            )
            
    if valid_loader is None:
        return history
    else:
        return history, valid_history

        
def metric_functions(scores: np.ndarray, target: np.ndarray) -> Dict:
    pred = scores.argmax(1)
    prob = softmax(scores, axis=1)
    K = scores.shape[1]
    onehot = np.eye(K)[target]
    
    def auroc_helper(x, y):
        try:
            return roc_auc_score(x, y, multi_class='ovr').tolist()
        except ValueError as e:
            print(e)
            return 0
            
    return {
        'conmat': confusion_matrix(target, pred).tolist(),
        'auroc': auroc_helper(onehot, scores),
        'kappa': cohen_kappa_score(target, pred, weights='linear').tolist(),
    }


def metrics_by_group(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset, 
        metric_functions: Callable,
    ) -> Dict[str, Dict[str, float]]:
    group_metrics = {}
    group_scores = collections.defaultdict(list)
    group_targets = collections.defaultdict(list)
    
    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        for i, (image, label, group, *_) in enumerate(dataset):
            image = image.to('cuda').unsqueeze(0)
            label = label[np.newaxis]
            score = model(image).detach().cpu().numpy()
            group_scores[group].append(score)
            group_targets[group].append(label)
            
    for group, scores in group_scores.items():
        scores = np.concatenate(group_scores[group])
        targets = np.concatenate(group_targets[group])
        metrics = metric_functions(scores, targets)
        group_metrics[group] = metrics
    
    return group_metrics

            
def main(args):
    debug = args.debug
    output_dir = pathlib.Path(args.output)
    if not debug:
        output_dir.mkdir()
        
    df = pd.read_csv(args.csv_file)
    
    patients = df.drop_duplicates('patient').patient
    test_df = df[df.patient.isin(patients.sample(frac=0.3))]
    train_df = df[~df.patient.isin(test_df.patient)]
    valid_patients = train_df.drop_duplicates('patient').sample(frac=0.2).patient
    valid_df = train_df[train_df.patient.isin(valid_patients)]
    train_df = train_df[~train_df.patient.isin(valid_patients)]
    
    for group in args.exclude:
        train_df = train_df.query('race != @group')
        valid_df = valid_df.query('race != @group')
        
    if debug:
        train_df = train_df.sample(n=256)
        valid_df = valid_df.sample(n=256)
        test_df = test_df.sample(n=256)
        
    if not debug:
        train_df.to_csv(output_dir / 'train_df.csv')
        valid_df.to_csv(output_dir / 'valid_df.csv')
        test_df.to_csv(output_dir / 'test_df.csv')
    
    if debug: print('df_shape'.rjust(20, '-'), df.shape)
    print('\tTRAIN\t', len(train_df))
    print('\tVALID\t', len(valid_df))
    print('\tTEST\t', len(test_df))
    train_ds = Dataset(df=train_df)
    valid_ds = Dataset(df=valid_df)
    test_ds = Dataset(df=test_df)
    
    dummy_image, *_ = train_ds[0]
    in_channels = dummy_image.shape[0]
    out_channels = df.label.unique().shape[0]
    if debug: print('image_shape'.rjust(20, '-'), dummy_image.shape)
    if debug: print('in_channels'.rjust(20, '-'), in_channels)
    if debug: print('out_channels'.rjust(20, '-'), out_channels)
    model = get_model(
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader_kwargs = {
        'shuffle': True,
        **loader_kwargs,
    }
    
    if args.use_equal_sampling:
        patients = train_df.drop_duplicates('patient').race.value_counts(normalize=True)
        weights = {k: 1 / v for k, v in patients.to_dict().items()}
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        weights = train_df.race.apply(lambda x: weights[x]).tolist()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, 
            num_samples=len(weights), 
            replacement=True,
        )
        train_loader_kwargs['sampler'] = sampler
        
    train_dl = get_loader(train_ds, **train_loader_kwargs)
    valid_dl = get_loader(valid_ds, **loader_kwargs)
    
    if args.weight_loss:
        weight = train_df.label.value_counts(normalize=True).to_dict()
        weight = {k: 1 / v for k, v in weight.items()}
        weight = {k: v / sum(weight.values()) for k, v in weight.items()}
        weight = torch.tensor(list(dict(sorted(weight.items())).values()))
        weight = weight.to('cuda')
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    history, valid_history = train_loop(
        model=model, 
        loader=train_dl,
        valid_loader=valid_dl,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learn_rate),
        criterion=criterion,
        epochs=args.epochs,
        metric_functions=metric_functions,
        debug=debug,
    )
    group_metrics = metrics_by_group(model, test_ds, metric_functions)
    
    if debug: 
        print(history)
        print(valid_history)
        print(group_metrics)
    else:
        with open(output_dir / 'history_train.json', 'w') as f:
            f.write(json.dumps(history))
        with open(output_dir / 'history_valid.json', 'w') as f:
            f.write(json.dumps(valid_history))
        with open(output_dir / 'group_test_metrics.json', 'w') as f:
            f.write(json.dumps(group_metrics))
        torch.save(model.state_dict(), output_dir / 'last.ckpt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='path to csv file')
    parser.add_argument('output', help='create directory')
    parser.add_argument(
        '-e',
        '--epochs', 
        type=int, 
        default=100,
        help='default number of epochs',
    )
    parser.add_argument(
        '-lr',
        '--learn_rate', 
        type=float, 
        default=0.0001,
        help='default learning rate',
    )
    parser.add_argument(
        '-bs',
        '--batch_size', 
        type=int, 
        default=16,
        help='default batch size',
    )
    parser.add_argument(
        '-nw',
        '--num_workers', 
        type=int, 
        default=8,
        help='default number of workers',
    )
    parser.add_argument(
        '-d',
        '--debug', 
        action='store_true',
        help='run with subset and use verbose printing',
    )
    parser.add_argument(
        '-w',
        '--weight_loss', 
        action='store_true',
        help='weight loss by inverse class frequency',
    )
    parser.add_argument(
        '--use_triplet_loss', 
        action='store_true',
        help='train with contrastive triplet loss',
    )
    parser.add_argument(
        '-eq',
        '--use_equal_sampling', 
        action='store_true',
        help='train with equal number of examples per group',
    )
    parser.add_argument(
        '--exclude', 
        nargs='+',
        choices=['ASIAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE'],
        default=['OTHER'],
        help='exclude these groups',
    )
    args = parser.parse_args()
    for k, v in args.__dict__.items():
        print('\t', k, ':\t', v)
        
    start = time.perf_counter()
    main(args=args)
    end = time.perf_counter()
    print('#' + 'DONE'.center(40, '=') + '#')
    print('#' + f'RUNTIME: {end - start:.0f}s'.center(40, '=') + '#')