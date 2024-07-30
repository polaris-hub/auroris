from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import datamol as dm
import numpy as np
import pandas as pd
from rdkit.Chem import FindMolChiralCenters

from auroris.curation.actions._base import BaseAction
from auroris.report import CurationReport
from auroris.types import VerbosityLevel
from auroris.visualization import visualize_chemspace


def curate_molecules(
    mols: List[Union[str, dm.Mol]],
    progress: bool = True,
    remove_stereo: bool = False,
    fix_mol: bool = True,
    count_stereoisomers: bool = True,
    count_stereocenters: bool = True,
    **parallelized_kwargs,
) -> Tuple:
    """
    Curate a list of molecules.

    Args:
        mols: List of molecules.
        progress: Whether show curation progress.
        fix_mol: Whether fix the error in molecule.
        remove_stereo: Whether remove stereo chemistry information from molecule.
        count_stereoisomers: Whether count the number of stereoisomers of molecule.
        count_stereocenters: Whether count the number of stereocenters of molecule.

    Returns:
        mol_dict: Dictionary of molecule and additional metadata
        num_invalid: Number of invßßalid molecules

    """
    fn = partial(
        _curate_molecule,
        remove_stereo=remove_stereo,
        fix_mol=fix_mol,
        count_stereoisomers=count_stereoisomers,
        count_stereocenters=count_stereocenters,
    )

    mol_list = dm.parallelized(
        fn=fn,
        inputs_list=mols,
        progress=progress,
        **parallelized_kwargs,
    )

    # Go from list of dicts to dict of lists
    mol_dict = {k: [dic[k] for dic in mol_list] for k in mol_list[0]}
    num_invalid = len([smi for smi in mol_dict["smiles"] if smi is None])
    return mol_dict, num_invalid


def _standardize_mol(mol: Union[dm.Mol, str], remove_stereo: bool = False, fix_mol: bool = False):
    """
    Standardize the molecular structure
    """
    # convert mol object
    mol = dm.to_mol(mol)

    # fix error in molecule using a greedy approach.
    if fix_mol:
        mol = dm.fix_mol(mol)

    # keep only the largest fragment, remove salt, metal etc.
    mol = dm.keep_largest_fragment(mol)

    # remove stereochemistry information
    if remove_stereo:
        mol = dm.remove_stereochemistry(mol)

    # standardize with/without stereochemistry information (tautomeric, protomeric forms)
    mol = dm.standardize_mol(
        mol=mol,
        disconnect_metals=False,
        reionize=True,
        normalize=True,
        uncharge=True,  # standardize for protomeric forms
        stereo=not remove_stereo,
    )

    return mol


def _curate_molecule(
    mol: Union[dm.Mol, str],
    remove_stereo: bool = False,
    fix_mol: bool = False,
    count_stereoisomers: bool = True,
    count_stereocenters: bool = True,
) -> dict:
    """
    Clean and standardize molecule to ensure the molecular structure.
    It comes with the option of remove salts/solvents and stereochemistry information from the molecule.

    Args:
        mol: Molecule
        remove_stero: Whether remove stereochemistry information from molecule
        count_stereoisomers: Whether count number of stereoisomers
        count_stereocenters: Whether count number of stereocenters

    Returns:
        mol_dict: Dictionary with the curated molecule and additional metadata
    """

    with dm.without_rdkit_log():
        try:
            mol = _standardize_mol(mol, remove_stereo=remove_stereo, fix_mol=fix_mol)
        except Exception:
            # The molecule could not be preprocessed. We assume it's an invalid molecule and return an empty dict
            return _get_mol_dict()

        smiles = dm.to_smiles(mol, canonical=True)
        molhash_id = dm.hash_mol(mol)
        molhash_id_no_stereo = dm.hash_mol(mol, hash_scheme="no_stereo")

        num_stereoisomers = None
        num_undefined_stereoisomers = None
        num_all_centers = None
        num_defined_centers = None
        num_undefined_centers = None
        undefined_e_d = None
        undefined_e_z = None

        if count_stereoisomers:
            # number of possible stereoisomers
            num_stereoisomers = dm.count_stereoisomers(
                mol=mol, undefined_only=False, rationalise=True, clean_it=True
            )

            # number of undefined stereoisomers
            num_undefined_stereoisomers = dm.count_stereoisomers(
                mol=mol, undefined_only=True, rationalise=True, clean_it=True
            )

        if count_stereocenters:
            # number of stereocenters
            num_all_centers, num_defined_centers, num_undefined_centers = _num_stereo_centers(mol)

            # None of the stereochemistry is defined in the molecule
            undefined_e_d = num_defined_centers == 0 and num_all_centers > 0

        if count_stereocenters and count_stereoisomers:
            # Undefined EZ stereochemistry which has no stereocenter.
            undefined_e_z = num_all_centers == 0 and num_undefined_centers

    return _get_mol_dict(
        smiles=smiles,
        molhash_id=molhash_id,
        molhash_id_no_stereo=molhash_id_no_stereo,
        num_stereoisomers=num_stereoisomers,
        num_undefined_stereoisomers=num_undefined_stereoisomers,
        num_defined_stereo_center=num_defined_centers,
        num_undefined_stereo_center=num_undefined_centers,
        num_stereo_center=num_all_centers,
        undefined_E_D=undefined_e_d,
        undefined_E_Z=undefined_e_z,
    )


def _get_mol_dict(
    smiles: Optional[str] = None,
    molhash_id: Optional[str] = None,
    molhash_id_no_stereo: Optional[str] = None,
    num_stereoisomers: Optional[int] = None,
    num_undefined_stereoisomers: Optional[int] = None,
    num_defined_stereo_center: Optional[int] = None,
    num_undefined_stereo_center: Optional[int] = None,
    num_stereo_center: Optional[int] = None,
    undefined_E_D: Optional[bool] = None,
    undefined_E_Z: Optional[bool] = None,
):
    return {
        "smiles": smiles,
        "molhash_id": molhash_id,
        "molhash_id_no_stereo": molhash_id_no_stereo,
        "num_stereoisomers": num_stereoisomers,
        "num_undefined_stereoisomers": num_undefined_stereoisomers,
        "num_defined_stereo_center": num_defined_stereo_center,
        "num_undefined_stereo_center": num_undefined_stereo_center,
        "num_stereo_center": num_stereo_center,
        "undefined_E_D": undefined_E_D,
        "undefined_E/Z": undefined_E_Z,
    }


def _num_stereo_centers(mol: dm.Mol) -> Tuple[int]:
    """
    Get the number of defined and undefined stereo centers of a given molecule
    by accessing the all and only defined stereo centers.
    It's to facilitate the analysis of the stereo isomers.
    None will be return if there is no stereo centers in the molecule.

     Args:
         mol: Molecule

    Returns:
        nun_defined_centers: Number of defined stereo centers.
        nun_undefined_centers: Number of undefined stereo centers.

    See Also:
        <rdkit.Chem.FindMolChiralCenters>

    """
    num_all_centers = len(FindMolChiralCenters(mol, force=True, includeUnassigned=True))
    if num_all_centers == 0:
        return 0, 0, 0

    num_defined_centers = len(FindMolChiralCenters(mol, force=True, includeUnassigned=False))
    nun_undefined_centers = num_all_centers - num_defined_centers
    return num_all_centers, num_defined_centers, nun_undefined_centers


class MoleculeCuration(BaseAction):
    """
    Automated molecule curation and chemistry space distribution.

    See [`auroris.curation.functional.curate_molecules`][] for the docs of the
    `remove_stereo`, `fix_mol`, `count_stereoisomers`, and `count_stereocenters` attributes

    Attributes:
        input_column: The name of the column that has the molecules (either `dm.Mol` objects or SMILES).
        X_col: Column with custom features for each of the molecules. If None, will use ECFP.
        y_cols: Column names for bioactivities, which will be used to colorcode the chemical space visualization.
    """

    name: Literal["mol_curation"] = "mol_curation"
    prefix: str = "MOL_"

    input_column: str
    remove_stereo: bool = False
    fix_mol: bool = False
    count_stereoisomers: bool = True
    count_stereocenters: bool = True

    X_col: Optional[str] = None
    y_cols: Optional[Union[str, List[str]]] = None

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        # Run the curation
        mols = dataset[self.input_column].values
        parallelized_kwargs = parallelized_kwargs or {}
        mol_dict, num_invalid = curate_molecules(
            mols,
            progress=verbosity > 1,
            remove_stereo=self.remove_stereo,
            fix_mol=self.fix_mol,
            count_stereoisomers=self.count_stereoisomers,
            count_stereocenters=self.count_stereocenters,
            **parallelized_kwargs,
        )
        mol_dict = {self.get_column_name(k): v for k, v in mol_dict.items()}
        df = pd.DataFrame(mol_dict)

        if num_invalid > 0:
            if report is not None:
                report.log(f"Couldn't preprocess {num_invalid} / {len(dataset)} molecules.")

        dataset = pd.concat([dataset, df], axis=1)

        # Log information to the report
        # - New columns with the curated molecule information

        if report is not None:
            for col in df.columns:
                report.log_new_column(col)

            smiles_col = self.get_column_name("smiles")
            smiles = dataset[smiles_col].dropna().values

            if self.X_col is None:
                featurizer = "ECFP"
                with dm.without_rdkit_log():
                    X = np.array([dm.to_fp(smi) for smi in smiles])
                report.log("Default `ecfp` fingerprint is used to visualize the chemical space.")

            else:
                featurizer = self.X_col
                X = dataset[self.X_col].values

            # list of data per column
            y = dataset[self.y_cols].T.values.tolist() if self.y_cols else None

            fig = visualize_chemspace(X=X, y=y, labels=self.y_cols)
            report.log_image(fig, title=f"Distribution in Chemical Space - {featurizer}")

            if self.count_stereocenters:
                # Plot all compounds with undefined stereocenters for visual inspection
                undefined_col = self.get_column_name("num_undefined_stereo_center")
                defined_col = self.get_column_name("num_defined_stereo_center")

                to_plot = dataset.query(f"{undefined_col} > 0")
                num_mol_undefined = to_plot.shape[0]
                report.log(f"Molecules with undefined stereocenter detected: {num_mol_undefined}.")

                if num_mol_undefined > 0:
                    legends = []
                    for _, row in to_plot.iterrows():
                        undefined = row[undefined_col]
                        defined = row[defined_col]
                        legends.append(f"Undefined:{undefined}\n Definded:{defined}")

                    image = dm.to_image(
                        to_plot[smiles_col].values, legends=legends, use_svg=False, returnPNG=True
                    ).data

                    report.log_image(
                        image,
                        title="Molecules with undefined stereocenters",
                        description=f"There are {num_mol_undefined} molecules with undefined stereocenter(s). "
                        "It's recommended to use <auroris.curation.action.StereoIsomerACDetection> and "
                        "check the stereoisomers and activity cliffs in the dataset.",
                    )

        return dataset
