import os
import os.path as osp
import re
from typing import Callable, Optional, List, Literal, Dict
from rdkit import Chem
import numpy as np
import torch
import shutil
from torch_geometric.data import InMemoryDataset, download_url, extract_gz, extract_zip
from torch_geometric.utils import from_smiles
import pandas as pd
import json
from tqdm import tqdm
from .features_generator import FeaturesGenerator


SMILES_TO_FEATURES: Dict[str, torch.tensor] = {}


class DrugExcpPair(InMemoryDataset):
    url = 'https://drive.google.com/uc?export=download&id=1W7H3PggeNXjIph5Hh0Jbedmrc6ow_w9S'

    def __init__(
        self,
        root: str,
        name: str = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{name}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['1440.csv', 'ext.csv'] + ['large_%d.csv' % i for i in range(10)]

    @property
    def processed_file_names(self) -> List[str]:
        return ['1440.pt', 'ext.pt'] + ['large_%d.pt' % i for i in range(10)]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self):
        fg = FeaturesGenerator(features_generator_name='rdkit_2d_normalized')
        fg_morgan = FeaturesGenerator(features_generator_name='morgan', num_bits=1024)
        for i, file in enumerate(self.raw_paths):
            data_list = []
            df = pd.read_csv(file)
            df['smiles'] = df['mixture'].apply(lambda x: json.loads(x)[0] + '.' + json.loads(x)[2])
            for j in tqdm(df.index, total=len(df)):
                data = from_smiles(df['smiles'][j])
                data.y = torch.tensor([df['class'][j]], dtype=torch.long).view(1, -1)
                data.smiles = df['smiles'][j].split('.')
                features = []
                for smiles in data.smiles:
                    if smiles not in SMILES_TO_FEATURES:
                        mol = Chem.MolFromSmiles(smiles)
                        fs = fg(mol)
                        fs = np.where(np.isnan(fs), 0, fs)
                        fs = np.concatenate([fs, fg_morgan(mol) - 0.5])
                        SMILES_TO_FEATURES[smiles] = torch.tensor(fs, dtype=torch.float32)
                    features.append(SMILES_TO_FEATURES[smiles])
                data.features = torch.cat(features).view(1, -1)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[i])

