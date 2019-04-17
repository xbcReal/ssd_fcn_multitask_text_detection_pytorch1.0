from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .icdar2015_dataset import icdar2015dataset
from .synthtext_dataset import synthtextdataset
from .icdar2013_dataset import icdar2013dataset

_DATASETS = {
    'icdar2015': icdar2015dataset
}


def build_dataset(dataset_list, transform=None, target_transform=None, is_test=False,args=None):
    datasets = []
    for dataset_name in dataset_list:
        print('dataset_name:',dataset_name)
        if dataset_name == 'icdar2015_train':
            args1 = dict()
            args1['transform'] = transform
            args1['target_transform'] = target_transform
            args1['keep_difficult'] = False
            args1['keep_difficult_in_score_map'] = False
            args1['filter_small_quad'] = False
            args1['thresh'] = 3.0 / 512
            dataset = icdar2015dataset(**args1)
            datasets.append(dataset)
        elif dataset_name == 'synthtext_train':
            args1 = dict()
            args1['transform'] = transform
            args1['target_transform'] = target_transform
            dataset = synthtextdataset(**args1)
            datasets.append(dataset)
        elif dataset_name == 'icdar2013_train':
            args1 = dict()
            args1['transform'] = transform
            args1['target_transform'] = target_transform
            dataset = icdar2013dataset(**args1)
            datasets.append(dataset)
        elif dataset_name == 'mlt2017_train':
            args1 = dict()
            args1['transform'] = transform
            args1['target_transform'] = target_transform
            dataset = icdar2013dataset(**args1)
            datasets.append(dataset)
        else:
            print('wrong dataset name')
            exit(233)
        # for testing, return a list of datasets
        if is_test:
            return datasets
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
            print('train_dataset:', dataset)
        elif len(datasets) == 1:
            dataset = datasets[0]
        else:
            exit(2233333)

    return dataset
