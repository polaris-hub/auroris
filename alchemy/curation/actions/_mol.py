from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import datamol as dm
import pandas as pd
from rdkit.Chem import FindMolChiralCenters

from alchemy.curation.actions._base import BaseAction
from alchemy.report import CurationReport
from alchemy.types import VerbosityLevel


def curate_molecules(
    mols: List[Union[str, dm.Mol]],
    progress: bool = True,
    remove_salt_solvent: bool = True,
    remove_stereo: bool = False,
    count_stereoisomers: bool = True,
    count_stereocenters: bool = True,
    **parallelized_kwargs,
):
    fn = partial(
        _curate_molecule,
        remove_salt_solvent=remove_salt_solvent,
        remove_stereo=remove_stereo,
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
    return mol_dict


def _standardize_mol(mol: Union[dm.Mol, str], remove_salt_solvent: bool = True, remove_stereo: bool = False):
    """
    Standardize the molecular structure
    """

    mol = dm.to_mol(mol)

    # fix mol
    mol = dm.fix_mol(mol)

    # sanitize molecule
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)

    if remove_salt_solvent:
        # standardize here to ensure the success the substructure matching for
        mol = dm.standardize_mol(
            mol=mol,
            disconnect_metals=False,
            reionize=True,
            normalize=True,
            uncharge=False,
            stereo=not remove_stereo,
        )
        # remove salts
        mol = dm.remove_salts_solvents(mol)

    # remove stereochemistry information
    if remove_stereo:
        mol = dm.remove_stereochemistry(mol)

    # standardize
    mol = dm.standardize_mol(
        mol=mol,
        disconnect_metals=False,
        reionize=True,
        normalize=True,
        uncharge=False,
        stereo=not remove_stereo,
    )

    return mol


def _curate_molecule(
    mol: Union[dm.Mol, str],
    remove_salt_solvent: bool = True,
    remove_stereo: bool = False,
    count_stereoisomers: bool = True,
    count_stereocenters: bool = True,
) -> dict:
    """
    Clean and standardize molecule to ensure the molecular structure.
    It comes with the option of remove salts/solvents and stereochemistry information from the molecule.

    Args:
        mol: Molecule

    Returns:
        mol_dict: Dictionary with the curated molecule and additional metadata
    """

    mol_dict = {}

    with dm.without_rdkit_log():
        mol = _standardize_mol(mol, remove_salt_solvent=remove_salt_solvent, remove_stereo=remove_stereo)

        mol_dict["smiles"] = dm.to_smiles(mol, canonical=True)
        mol_dict["molhash_id"] = dm.hash_mol(mol)
        mol_dict["molhash_id_no_stereo"] = dm.hash_mol(mol, hash_scheme="no_stereo")

        if count_stereoisomers:
            # number of possible stereoisomers
            mol_dict["num_stereoisomers"] = dm.count_stereoisomers(
                mol=mol, undefined_only=False, rationalise=True, clean_it=True
            )

            # number of undefined stereoisomers
            mol_dict["num_undefined_stereoisomers"] = dm.count_stereoisomers(
                mol=mol, undefined_only=True, rationalise=True, clean_it=True
            )

        if count_stereocenters:
            # number of stereocenters
            num_all_centers, num_defined_centers, num_undefined_centers = _num_stereo_centers(mol)

            # number of undefined stereoisomers
            mol_dict["num_defined_stereo_center"] = num_defined_centers
            mol_dict["num_undefined_stereo_center"] = num_undefined_centers
            mol_dict["num_stereo_center"] = num_all_centers

            # None of the stereochemistry is defined in the molecule
            mol_dict["undefined_E_D"] = num_defined_centers == 0 and num_all_centers > 0

        if count_stereocenters and count_stereoisomers:
            # Undefined EZ stereochemistry which has no stereocenter.
            mol_dict["undefined_E_D"] = (
                mol_dict["num_stereo_center"] == 0 and mol_dict["num_undefined_stereoisomers"] > 0,
            )

    return mol_dict


def _num_stereo_centers(mol: dm.Mol) -> Tuple[int]:
    """Get the number of defined and undefined stereo centers of a given molecule
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
    Args:
        input_column: The name of the column that has the molecules (either `dm.Mol` objects or SMILES).
        remove_salt_solvent: When set to 'True', all disconnected salts and solvents
            will be removed from molecule. In most of the cases, it is recommended to remove the salts/solvents.
        remove_stereo: Whether remove stereochemistry information from molecule.
            If it's known that the stereochemistry do not contribute to the bioactivity of interest,
            the stereochemistry information can be removed.
    """

    input_column: str
    prefix: str = "MOL_"
    remove_salt_solvent: bool = True
    remove_stereo: bool = False
    count_stereoisomers: bool = True
    count_stereocenters: bool = True

    def transform(
        self,
        dataset: pd.DataFrame,
        report: Optional[CurationReport] = None,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        parallelized_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        mols = dataset[self.input_column].values

        parallelized_kwargs = parallelized_kwargs or {}
        mol_dict = curate_molecules(
            mols,
            progress=verbosity > 1,
            remove_salt_solvent=self.remove_salt_solvent,
            remove_stereo=self.remove_stereo,
            count_stereoisomers=self.count_stereoisomers,
            count_stereocenters=self.count_stereocenters,
            **parallelized_kwargs,
        )
        mol_dict = {self.get_column_name(k): v for k, v in mol_dict.items()}
        df = pd.DataFrame(mol_dict)

        if report is not None:
            for col in df.columns:
                report.log(f"Added the {col} column.")

        dataset = pd.concat([dataset, df], axis=1)
        return dataset
