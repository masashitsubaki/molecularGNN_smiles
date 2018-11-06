from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem


def create_atoms(mol):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atoms)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        return np.array(atoms)

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints
        return np.array(fingerprints)


def pickle_dump(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    DATASET, radius = sys.argv[1:]
    radius = int(radius)

    with open('../../dataset/classification/' + DATASET +
              '/original/smiles_property.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude the data contains "." in the smiles."""
    data_list = list(filter(lambda x:
                     '.' not in x.strip().split()[0], data_list))
    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    Molecules, Adjacencies, Properties = [], [], []

    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

        smiles, property = data.strip().split()

        mol = Chem.MolFromSmiles(smiles)
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
        Molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        Adjacencies.append(adjacency)

        property = np.array([int(property)])
        Properties.append(property)

    dir_input = ('../../dataset/classification/' + DATASET +
                 '/input/radius' + str(radius) + '/')
    if not os.path.isdir(dir_input):
        os.mkdir(dir_input)
    np.save(dir_input + 'molecules', Molecules)
    np.save(dir_input + 'adjacencies', Adjacencies)
    np.save(dir_input + 'properties', Properties)

    if (radius == 0):
        fingerprint_dict = atom_dict
    pickle_dump(fingerprint_dict, dir_input + 'fingerprint_dict' + '.pickle')

    print('The preprocess has finished!')
