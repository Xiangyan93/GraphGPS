from tap import Tap
from typing import List, Literal
import pandas as pd
from sklearn.model_selection import KFold
from mgktools.data.split import data_split_index


class TrainArgs(Tap):
    cfg_file: str
    """The configuration file path for the GPS model."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    cross_validation: Literal["n-fold", "Monte-Carlo"] = "Monte-Carlo"
    """The way to split data for cross-validation."""
    nfold: int = None
    """The number of fold for n-fold CV."""
    split_type: Literal["random", "scaffold_order", "scaffold_random", "stratified"] = None
    """Method of splitting the data into train/test sets."""
    split_sizes: List[float] = None
    """Split proportions for train/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """Random seed."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    separate_val_path: str = None
    """Path to separate validation set, optional."""
    output_details: bool = False
    """output the exact prediction for every epoch."""

    def process_args(self) -> None:
        self.opts = ['wandb.use', 'False']
        self.split_index_list = []
        df = pd.read_csv(self.data_path)
        if self.separate_test_path is not None:
            df_test = pd.read_csv(self.separate_test_path)
            if self.separate_val_path is None:
                self.split_index_list.append([df.index.tolist(), [], (df_test.index + len(df)).tolist()])
            else:
                df_val = pd.read_csv(self.separate_val_path)
                self.split_index_list.append([df.index.tolist(), (df_val.index + len(df)).tolist(), (df_test.index + len(df) + len(df_val)).tolist()])
        else:
            for i in range(self.num_folds):
                if self.cross_validation == 'Monte-Carlo':
                    assert self.split_type is not None, 'split_type must be specified for Monte-Carlo cross-validation.'
                    assert self.split_sizes is not None, 'split_sizes must be specified for Monte-Carlo cross-validation.'
                    if self.split_type == 'scaffold_order' or self.split_type == 'scaffold_random':
                        assert len(self.smiles_columns) == 1, 'Only one SMILES column is allowed for scaffold splitting.'
                        mols = df[self.smiles_columns[0]].tolist()
                    else:
                        mols = None
                    split_index = data_split_index(n_samples=len(df), 
                                                mols=mols,
                                                targets=df[self.target_columns].to_numpy().ravel(),
                                                split_type=self.split_type, 
                                                sizes=self.split_sizes, 
                                                seed=self.seed + i)
                    self.split_index_list.append(split_index)
                elif self.cross_validation == 'n-fold':
                    assert self.nfold is not None, 'nfold must be specified for nfold cross-validation.'
                    kf = KFold(n_splits=self.nfold, shuffle=True, random_state=self.seed + i)
                    kf.get_n_splits(df[self.smiles_columns])
                    for i_fold, (train_index, test_index) in enumerate(kf.split(df[self.smiles_columns])):
                        split_index = [train_index, [], test_index]
                        self.split_index_list.append(split_index)
                else:
                    raise ValueError(f'cross_validation "{self.cross_validation}" not supported.')


class PredictArgs(Tap):
    cfg_file: str
    """The configuration file path for the GPS model."""
    checkpoint_dir: str
    """Directory from which to load model checkpoints."""
    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    def process_args(self) -> None:
        self.opts = ['wandb.use', 'False']
        