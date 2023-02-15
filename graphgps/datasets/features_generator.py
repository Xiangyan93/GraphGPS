#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List, Union
import math
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


class FeaturesGenerator:
    def __init__(self, features_generator_name: Union[str, Callable],
                 radius: int = 2,
                 num_bits: int = 2048):
        self.features_generator_name = features_generator_name
        self.radius = radius
        self.num_bits = num_bits
        if features_generator_name in ['morgan', 'morgan_count']:
            assert self.radius is not None
            assert self.num_bits is not None

    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        if self.features_generator_name.__class__ != str:
            return self.features_generator_name(mol)
        if self.features_generator_name == 'morgan':
            return self.morgan_binary_features_generator(mol)
        elif self.features_generator_name == 'morgan_count':
            return self.morgan_counts_features_generator(mol)
        elif self.features_generator_name == 'rdkit_208':
            return self.rdkit_208_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d':
            return self.rdkit_2d_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d_normalized':
            return self.rdkit_2d_normalized_features_generator(mol)
        else:
            raise ValueError(f'unknown features generator: {self.features_generator_name}')

    def morgan_binary_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a binary Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the binary Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features

    def morgan_counts_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a counts-based Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the counts-based Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features

    @staticmethod
    def rdkit_2d_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @staticmethod
    def rdkit_2d_normalized_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D normalized features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features

    @staticmethod
    def rdkit_208_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        # define chemical features for molecular descriptions
        descr = Descriptors._descList
        calc = [x[1] for x in descr]
        ds_n = []
        for d in calc:
            v = d(mol)
            if v > np.finfo(np.float32).max:  # postprocess descriptors for freak large values
                ds_n.append(np.finfo(np.float32).max)
            elif math.isnan(v):
                ds_n.append(np.float32(0.0))
            else:
                ds_n.append(np.float32(v))
        return np.array(ds_n)
