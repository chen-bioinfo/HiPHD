import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict
import json
import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import data, core, utils, datasets, tasks, models, transforms
from torchdrug.layers.geometry import AlphaCarbonNode
from torchdrug.utils import comm
from torchdrug.core import Engine
from rdkit import Chem


def extract_hb_raw(pdb_file):
    pdb_file = os.path.expanduser(pdb_file)
    dssp = os.path.join(os.path.dirname(__file__), 'dssp')
    
    dssp_file = os.path.splitext(pdb_file)[0]+'.dssp'
    if not os.path.exists(dssp_file):
        os.system(' '.join([dssp,'-i', pdb_file, '-o', dssp_file]))
    if not os.path.isfile(dssp_file):
        return None
    with open(dssp_file, "r") as f:
        dssp_data = f.readlines()[28:]
    data = {"model":{}}
    for line in dssp_data:
        idx = int(line[0:5])
        data["model"][str(idx)] = {
                "res": line[13],
                "ss": line[16],
                "acc": int(line[34:38]),
                "nho0p": int(line[39:45]),
                "nho0e": float(line[46:50]),
                "ohn0p": int(line[50:56]),
                "ohn0e": float(line[57:61]),
                "nho1p": int(line[61:67]),
                "nho1e": float(line[68:72]),
                "ohn1p": int(line[72:78]),
                "ohn1e": float(line[79:83]),
                "phi": float(line[103:109]),
                "psi": float(line[109:115]),
                "x": float(line[115:122]),
                "y": float(line[122:129]),
                "z": float(line[129:136]),
        }
    json_file = dssp_file.replace(".dssp", ".json")
    with open(json_file, "w") as f:
        json.dump(data, f)
    return data

def cal_hbond(model_dict):
    residue_name = list('LAGVESKIDTRPNFQYHMCW')
    radius = 8
    cutoff = 4
    hbondmax = 4
    hb_types = {k:v for k,v in zip(
            [*range(-5, -2), *range(3, 6)], range(6)
        )}
    
    model = model_dict['model']
    size = len(model)
    coord = []
    for _,v in model.items():
        coord.append([v['x'], v['y'], v['z']])
    coord = torch.as_tensor(coord)
    dist2 = torch.cdist(coord.unsqueeze(0), coord.unsqueeze(0)).squeeze().numpy()
    dist2 = torch.as_tensor(dist2)
    dist2 = torch.nan_to_num(dist2, nan=torch.inf)

    # idx/model is in [1, size]
    # dist in in [0,size)
    # type of idx is int
    is_frag = lambda idx: all((i > 0 and i < size) 
                                and model[str(i)]['res'] in residue_name
                                and dist2[i-1, i] <= cutoff
                            for i in range(idx-radius-1, idx+radius+1))
    edge_list = []
    for idx, res in model.items():
        idx = int(idx)
        if (res['nho0e']) <= hbondmax:
            idx1 = idx + (res['nho0p'])
            if abs(idx - idx1) > 2 and is_frag(idx) and is_frag(idx1):
                if res['nho0p'] in hb_types:
                    hb_type = hb_types[res['nho0p']]
                else:
                    hb_type = len(hb_types)
                edge_list.append([idx-1, idx1-1, hb_type])# idx of graph is in [0,size)
        
        if (res['nho1e']) <= hbondmax:
            idx2 = idx + (res['nho1p'])
            if abs(idx - idx2) > 2 and is_frag(idx) and is_frag(idx2):
                if res['nho1p'] in hb_types:
                    hb_type = hb_types[res['nho1p']]
                else:
                    hb_type = len(hb_types)
                edge_list.append([idx-1, idx2-1, hb_type])
    if len(edge_list) < 20:
        return None
    edge_list = torch.as_tensor(edge_list)
    return edge_list

def load_pdb(pdb_file, verbose = 0):
    hb_type = {k:v for k,v in zip(
        [*range(-5, -2), *range(3, 6)], range(6)
    )}

    mol = Chem.MolFromPDBFile(pdb_file)
    if not mol:
        logging.debug("RDKit cannot read PDB file `%s`" % pdb_file)
        return None
    prot = data.Protein.from_molecule(mol)
    if not prot:
        logging.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
        return None
    if prot.num_residue < 20:
        logging.debug("Protein is too short from pdb file `%s`. Ignore this sample." % pdb_file)
        return None
    if prot.num_residue > 1022:
        logging.debug("Protein is too long from pdb file `%s`. Ignore this sample." % pdb_file)
        return None
    
    prot = AlphaCarbonNode().forward(prot)
    # hbond_file = str.replace(pdb_file, 'pdb', 'json')
    # if not os.path.exists(hbond_file):
    #     hbond_raw = extract_hb_raw(pdb_file)
    #     if hbond_raw is None:
    #         logging.debug("Fail to extract HBond from pdb file `%s`. Ignore this sample." % pdb_file)
    #         return None
    # else:
    #     with open(hbond_file, 'r') as fp:
    #         hbond_raw = json.load(fp)
    hbond_raw = extract_hb_raw(pdb_file)
    if hbond_raw is None:
        logging.debug("Fail to extract HBond from pdb file `%s`. Ignore this sample." % pdb_file)
        return None

    # Now Analyze HBonds
    edge_list = cal_hbond(hbond_raw)
    if edge_list is None:
        logging.debug("No enough HBond from pdb file `%s`. Ignore this sample." % pdb_file)
        return None
    # Finish Analyzing
    # edge_list = torch.as_tensor([[0, 0, 0]])
    bond_type = edge_list[:,2]
    num_relation = torch.as_tensor(len(hb_type)+1, device=prot.device)
    
    protein = data.Protein(edge_list, prot.atom_type, bond_type, num_node=prot.num_atom, num_residue=prot.num_residue,
                            node_position=prot.node_position, atom_name=prot.atom_name, num_relation = num_relation,
                            atom2residue=prot.atom2residue, residue_feature=prot.residue_feature, 
                            residue_type=prot.residue_type)
    item = {'graph' : protein}
    item = transforms.ProteinView('residue')(item)
    return item

