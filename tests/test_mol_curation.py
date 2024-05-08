import datamol as dm

from auroris.curation.actions._mol import _num_stereo_centers
from auroris.curation.functional import curate_molecules


def test_run_chemistry_curation():
    mols = [
        "COc1ccc2ncc(=O)n(CCN3CC[C@H](NCc4ccc5c(n4)NC(=O)CO5)[C@H](O)C3)c2c1",
        "COc1ccc2ncc(=O)n(CCN3CC[C@@H](NCc4ccc5c(n4)NC(=O)CO5)[C@@H](O)C3)c2c1",
        "C[C@H]1CN(Cc2cc(Cl)ccc2OCC(=O)O)CCN1S(=O)(=O)c1ccccc1",
        "C[C@@H]1CN(Cc2cc(Cl)ccc2OCC(=O)O)CCN1S(=O)(=O)c1ccccc1",
        "CC[C@@H](c1ccc(C(=O)O)c(Oc2cccc(Cl)c2)c1)N1CCC[C@H](n2cc(C)c(=O)[nH]c2=O)C1",
        "CC[C@H](c1ccc(C(=O)O)c(Oc2cccc(Cl)c2)c1)N1CCC[C@H](n2cc(C)c(=O)[nH]c2=O)C1",
        "CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12",
        "CCOc1cc2nn(CCC(C)(C)O)cc2cc1NC(=O)c1cccc(C(F)F)n1",
        "CN(c1ncc(F)cn1)C1CCCNC1",
        "CC(C)(Oc1ccc(-c2cnc(N)c(-c3ccc(Cl)cc3)c2)cc1)C(=O)O",
        "CC(C)(O)CCn1cc2cc(NC(=O)c3cccc(C(F)(F)F)n3)c(C(C)(C)O)cc2n1",
        "CC#CC(=O)N[C@H]1CCCN(c2c(F)cc(C(N)=O)c3[nH]c(C)c(C)c23)C1",
        "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
        "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        "CC(C)NC(=O)COc1cccc(-c2nc(Nc3ccc4[nH]ncc4c3)c3ccccc3n2)c1.[Na]",
        "CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1.Cl",
    ]

    # check stereoisomers are included.
    mol_dict, _ = curate_molecules(mols=mols)
    unique_smiles = set(mol_dict["smiles"])
    assert len(unique_smiles) == len(mols)

    # check if stereoisomers are ignored
    mol_dict, _ = curate_molecules(mols=mols, remove_stereo=True)
    unique_smiles = set(mol_dict["smiles"])
    assert len(unique_smiles) == len(mols) - 3

    # check whether salts/solvents were removed.
    for smiles in mol_dict["smiles"]:
        mol = dm.to_mol(smiles)
        assert dm.same_mol(dm.remove_salts_solvents(mol), mol)


def test_num_undefined_stereo_centers():
    # mol with no stereo centers
    mol = dm.to_mol("CCCC")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 0
    assert num_defined == 0
    assert num_undefined == 0

    # mol with all defined centers
    mol = dm.to_mol("C1C[C@H](C)[C@H](C)[C@H](C)C1")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 3
    assert num_defined == 3
    assert num_undefined == 0

    # mol with partial defined centers
    mol = dm.to_mol("C[C@H](F)C(F)(Cl)Br")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 2
    assert num_defined == 1
    assert num_undefined == 1

    # mol with no defined centers
    mol = dm.to_mol("CC(F)C(F)(Cl)Br")
    num_all, num_defined, num_undefined = _num_stereo_centers(mol)
    assert num_all == 2
    assert num_defined == 0
    assert num_undefined == 2
