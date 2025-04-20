from typing import Callable, Optional, List, Literal, Dict
import os
import os.path as osp
import shutil
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.datasets.molecule_net import MoleculeNet
from torch_geometric.utils import from_smiles
from mgktools.features_mol.features_generators import FeaturesGenerator


SMILES_TO_FEATURES: Dict[str, torch.tensor] = {}


class DatasetFromCSVFile(InMemoryDataset):
    def __init__(
        self,
        data_path: str,
        smiles_columns: List[str],
        target_columns: List[str] = None,
        features_generator: List[str] = None,
        task_type: Literal['classification', 'regression'] = 'regression',
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        self.data_path = data_path
        self.file_name = data_path.split('/')[-1].split('.')[0]
        self.smiles_columns = smiles_columns
        self.target_columns = target_columns
        self.features_generator = features_generator
        self.task_type = task_type
        super(DatasetFromCSVFile, self).__init__(root, transform, pre_transform, pre_filter, log)
        path = osp.join(self.processed_dir, self.file_name + '.pt')
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.file_name + '.csv']

    @property
    def processed_file_names(self) -> List[str]:
        return [self.file_name + '.pt']

    def download(self):
        # copy file from data_path to root
        shutil.copyfile(self.data_path, osp.join(self.raw_dir, self.file_name + '.csv'))

    def process(self):
        fgs = [FeaturesGenerator(features_generator_name=fg) for fg in self.features_generator] if self.features_generator is not None else []
        for i, file in enumerate(self.raw_paths):
            data_list = []
            df = pd.read_csv(file)
            for smiles_column in self.smiles_columns:
                assert smiles_column in df.columns
            df['smiles'] = df.apply(lambda x: '.'.join([x[smiles_column] for smiles_column in self.smiles_columns]), axis=1)
            for j in tqdm(df.index, total=len(df)):
                data = from_smiles(df['smiles'][j])
                # set aromatic bonds from 12 to 4, otherwise the bond embedding will be out of range.
                mask = data.edge_attr[:, 0] == 12
                data.edge_attr[mask, 0] = 4
                if self.target_columns is None:
                    data.y = None
                elif self.task_type == 'regression':
                    data.y = torch.tensor([df[self.target_columns].to_numpy()[j].tolist()], dtype=torch.float32).view(1, -1)
                else:
                    data.y = torch.tensor([df[self.target_columns].to_numpy()[j].tolist()], dtype=torch.int32).view(1, -1)
                data.smiles = df[self.smiles_columns].iloc[j].tolist()
                if len(fgs) > 0:
                    features = []
                    for smiles in data.smiles:
                        if smiles not in SMILES_TO_FEATURES:
                            mol = Chem.MolFromSmiles(smiles)
                            features_mol = []
                            for fg in fgs:
                                fs = fg(mol)
                                fs = np.where(np.isnan(fs), 0, fs)
                                features_mol.append(fs)
                            features_mol = np.concatenate(features_mol)
                            SMILES_TO_FEATURES[smiles] = torch.tensor(features_mol, dtype=torch.float32)
                        features.append(SMILES_TO_FEATURES[smiles])
                    data.features = torch.cat(features).view(1, -1)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[i])
