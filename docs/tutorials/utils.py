from typing import Callable

from rdkit import Chem
from rdkit.Chem import AllChem

import datamol as dm
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def add_3d_coord(mol: dm.Mol):
    """Add 3d coordinates to molecule"""
    try:
        # convert to mol object
        mol = dm.to_mol(mol)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)

        # Optimize the 3D structure
        AllChem.UFFOptimizeMolecule(mol)
        return mol
    except Exception as e:
        print(e)
        return None


def evaluate(
    data_test: pd.DataFrame,
    transformer: Callable,
    model: RandomForestClassifier,
    mol_col: str,
    val_col: str,
):
    """Evaluate on test set with classification metrics"""
    X_test = transformer(data_test[mol_col].values)
    y_test = data_test[val_col].values

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report["macro avg"]
