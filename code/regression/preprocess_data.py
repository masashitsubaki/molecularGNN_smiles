from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuple of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    DATASET, radius = sys.argv[1:]
    radius = int(radius)

    with open('../../dataset/regression/' + DATASET +
              '/original/data.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude data contains "." in its smiles."""
    data_list = list(filter(lambda x:
                     '.' not in x.strip().split()[0], data_list))
    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    Smiles, molecules, adjacencies, properties = '', [], [], []

    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

        smiles, property = data.strip().split()
        Smiles += smiles + '\n'

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # AddHs.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        properties.append([[float(property)]])

    """Normalize properties (i.e., mean 0 and std 1)."""
    p = np.array(properties)
    mean, std = np.mean(p), np.std(p)
    properties = np.array((p - mean) / std)

    dir_input = ('../../dataset/regression/' + DATASET +
                 '/input/radius' + str(radius) + '/')
    os.makedirs(dir_input, exist_ok=True)

    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_input + 'molecules', molecules)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'properties', properties)
    np.save(dir_input + 'mean', mean)
    np.save(dir_input + 'std', std)
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

    print('The preprocess has finished!')
