import os
import glob
import h5py
import torch
import warnings
import json
import logging

from rdkit import Chem
from tqdm import tqdm

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R
from torchdrug.layers.geometry import AlphaCarbonNode
import subprocess
logger = logging.getLogger(__name__)

@R.register("datasets.CATH")
class CATH(data.ProteinDataset):
    processed_file = "CATH.pkl.gz"
    splits = ["train", "valid", "test"]
    lab2idx_file = 'lab2idx.json'
    metainfo_file = 'all.json'
    split_file = 'split.json'

    def __init__(self, path,multitask = False ,verbose=1,**kwargs):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.multitask = multitask

        pkl_file = os.path.join(self.path, self.processed_file)
        map_file = os.path.join(self.path, self.lab2idx_file)
        label_file = os.path.join(self.path, self.metainfo_file)
        split_file = os.path.join(self.path, self.split_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = glob.glob(os.path.join(self.path, 'dompdb', '*'))
            self.load_pdbs(pdb_files)
            self.save_pickle(pkl_file)
        self.name = [os.path.basename(single) for single in self.pdb_files]

        # adjust the order
        raw_split =  json.load(open(split_file, 'r'))
        for k in raw_split:
            raw_split[k] = [s for s in raw_split[k] if s in self.name]        
        split_list = [single for single_split in raw_split.values() for single in single_split ]
        name_order = {k:i for i,k in enumerate(split_list)}

        # name, pdb_files, data, sequences = [],[],[],[]
        # for i in range(len(self.name)):
        #     idx = name_order[self.name[i]]
        #     name[i], pdb_files[i], data[i], sequences[i] = \
        #     self.name[idx], self.pdb_files[idx], self.data[idx], self.sequences[idx]
        # self.name, self.pdb_files, self.data, self.sequences = \
        #     name, pdb_files, data, sequences
        indices = [name_order[name] for name in self.name]
        self.name = [self.name[idx] for idx in indices]
        self.pdb_files = [self.pdb_files[idx] for idx in indices]
        self.data = [self.data[idx] for idx in indices]
        self.sequences = [self.sequences[idx] for idx in indices]

        class_map = json.load(open(map_file, 'r'))
        label_list =  json.load(open(label_file))
        self.class_map = class_map
        self.label_list = label_list

        # print(class_map.keys())
        fold_labels = [
            [
                class_map['c'][label_list[single]['c']],
                class_map['a'][label_list[single]['a']],
                class_map['t'][label_list[single]['t']],
                class_map['h'][label_list[single]['h']],
            ]
            for single in self.name
        ]
        fold_labels = torch.as_tensor(fold_labels)
        if self.multitask:
            self.targets = {
                'c': [s[0] for s in fold_labels], 
                'a': [s[1] for s in fold_labels], 
                't':[s[2] for s in fold_labels],
                'h':[s[3] for s in fold_labels]
                }
        else:
            self.targets = {
                'h':[s[3] for s in fold_labels]
                }

        self.num_samples = []
        for split in self.splits:            
            self.num_samples.append(len(raw_split[split]))

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0):
        self.transform = transform
        self.lazy = lazy
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                try:
                    mol = Chem.MolFromPDBFile(pdb_file)
                    if not mol: continue
                    protein = data.Protein.from_molecule(mol)
                    if not protein: continue
                except:
                    continue
            else:
                protein = None
            if protein.num_residue < 20 or protein.num_residue > 1022:
                continue
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

    def split(self):
        keys = self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        # print('Len of split is %d'%len(splits))
        return splits

    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # item = {"graph": protein, "h": self.targets["h"][index]}
        if self.multitask:
            item = {"graph": protein, 
                    "c": self.targets["c"][index],
                    "a":self.targets['a'][index],
                    't':self.targets['t'][index],
                    "h": self.targets["h"][index]}
        else:
            item = {"graph": protein, 
                    "h": self.targets["h"][index]}

        if self.transform:
            item = self.transform(item)
        return item
@R.register("datasets.HBCATH")
class HBCATH(data.ProteinDataset):
    # processed_file = "scopehb.pkl.gz"
    processed_file = "HBCATH.pkl.gz"
    splits = ["train", "valid","test"]
    lab2idx_file = 'lab2idx.json'
    metainfo_file = 'all.json'
    split_file = 'split.json'

    def __init__(self, path,radius = 8 ,cutoff=4,verbose=1,hbondmax = 4,multitask = False,**kwargs):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.residue_name = list('LAGVESKIDTRPNFQYHMCW')
        self.radius = radius
        self.cutoff = cutoff
        self.hbondmax = hbondmax
        self.hb_type = {k:v for k,v in zip(
            [*range(-5, -2), *range(3, 6)], range(6)
        )} 
        self.multitask = multitask

        pkl_file = os.path.join(self.path, self.processed_file)
        map_file = os.path.join(self.path, self.lab2idx_file)
        label_file = os.path.join(self.path, self.metainfo_file)
        split_file = os.path.join(self.path, self.split_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = glob.glob(os.path.join(self.path, 'dompdb', '*'))
            self.load_pdbs(pdb_files)
            self.save_pickle(pkl_file)
        self.name = [os.path.basename(single) for single in self.pdb_files]
        # adjust the order
        raw_split =  json.load(open(split_file, 'r'))
        for k in raw_split:
            raw_split[k] = [s for s in raw_split[k] if s in self.name]
        split_list = [single for single_split in raw_split.values() for single in single_split ]
        name_order = {k:i for i,k in enumerate(split_list)}

        indices = [name_order[name] for name in self.name]
        self.name = [self.name[idx] for idx in indices]
        self.pdb_files = [self.pdb_files[idx] for idx in indices]
        self.data = [self.data[idx] for idx in indices]
        self.sequences = [self.sequences[idx] for idx in indices]

        class_map = json.load(open(map_file, 'r'))
        label_list =  json.load(open(label_file))
        self.class_map = class_map
        self.label_list = label_list

        fold_labels = [
            [
                class_map['c'][label_list[single]['c']],
                class_map['a'][label_list[single]['a']],
                class_map['t'][label_list[single]['t']],
                class_map['h'][label_list[single]['h']],
            ]
            for single in self.name
        ]
        fold_labels = torch.as_tensor(fold_labels)
        if self.multitask:
            self.targets = {
                'c': [s[0] for s in fold_labels], 
                'a': [s[1] for s in fold_labels], 
                't':[s[2] for s in fold_labels],
                'h':[s[3] for s in fold_labels]
                }
        else:
            self.targets = {
                'h':[s[3] for s in fold_labels]
                }


        self.num_samples = []
        for split in self.splits:            
            self.num_samples.append(len(raw_split[split]))
        
    def extract_hb_raw(self, pdb_file):
        dssp = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )), 'dssp')
        
        dssp_file = pdb_file+'.dssp'
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

    def cal_hbond(self, model_dict):
        residue_name = self.residue_name
        radius = self.radius
        cutoff = self.cutoff
        hbondmax = self.hbondmax

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
                    if res['nho0p'] in self.hb_type:
                        hb_type = self.hb_type[res['nho0p']]
                    else:
                        hb_type = len(self.hb_type)
                    edge_list.append([idx-1, idx1-1, hb_type])# idx of graph is in [0,size)
            
            if (res['nho1e']) <= hbondmax:
                idx2 = idx + (res['nho1p'])
                if abs(idx - idx2) > 2 and is_frag(idx) and is_frag(idx2):
                    if res['nho1p'] in self.hb_type:
                        hb_type = self.hb_type[res['nho1p']]
                    else:
                        hb_type = len(self.hb_type)
                    edge_list.append([idx-1, idx2-1, hb_type])
        if len(edge_list) < 20:
            return None
        edge_list = torch.as_tensor(edge_list)
        return edge_list

    def load_pdb(self, pdb_file, verbose = 0):
        mol = Chem.MolFromPDBFile(pdb_file)
        if not mol:
            logger.debug("RDKit cannot read PDB file `%s`" % pdb_file)
            return None
        prot = data.Protein.from_molecule(mol)
        if not prot:
            logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        if prot.num_residue < 20:
            logger.debug("Protein is too short from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        if prot.num_residue > 1022:
            logger.debug("Protein is too long from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        
        prot = AlphaCarbonNode().forward(prot)
        hbond_file = pdb_file+'.json'
        if not os.path.exists(hbond_file):
            hbond_raw = self.extract_hb_raw(pdb_file)
            if hbond_raw is None:
                logger.debug("Fail to extract HBond from pdb file `%s`. Ignore this sample." % pdb_file)
                return None
        else:
            with open(hbond_file, 'r') as fp:
                hbond_raw = json.load(fp)

        # Now Analyze HBonds
        edge_list = self.cal_hbond(hbond_raw)
        if edge_list is None:
            logger.debug("No enough HBond from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        # Finish Analyzing
        # edge_list = torch.as_tensor([[0, 0, 0]])
        bond_type = edge_list[:,2]
        num_relation = torch.as_tensor(len(self.hb_type)+1, device=prot.device)
        
        protein = data.Protein(edge_list, prot.atom_type, bond_type, num_node=prot.num_atom, num_residue=prot.num_residue,
                               node_position=prot.node_position, atom_name=prot.atom_name, num_relation = num_relation,
                                atom2residue=prot.atom2residue, residue_feature=prot.residue_feature, 
                                residue_type=prot.residue_type)
        return protein
        

    
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0):
        # num_sample = len(pdb_files)
        # if num_sample > 1000000:
        #     warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
        #                 "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                try:    protein = self.load_pdb(pdb_file)
                except: 
                    continue
                if protein is None:
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

        

    def split(self):
        keys = self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        # print('Len of split is %d'%len(splits))
        return splits


    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            try: protein = self.data[index].clone()
            except: print(index)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # Superfamily ONLY
        if self.multitask:
            item = {"graph": protein, 
                    "c": self.targets["c"][index],
                    "a":self.targets['a'][index],
                    't':self.targets['t'][index],
                    "h": self.targets["h"][index]}
        else:
            item = {"graph": protein, 
                    "h": self.targets["h"][index]}

        if self.transform:
            item = self.transform(item)
        return item


@R.register("datasets.Scope")
class Scope(data.ProteinDataset):
    processed_file = "scope.pkl.gz"
    splits = ["train", "valid", "test"]
    lab2idx_file = 'lab2idx.json'
    metainfo_file = 'all.json'

    def __init__(self, path, multitask = False,verbose=1,**kwargs):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.multitask = multitask

        pkl_file = os.path.join(self.path, self.processed_file)
        map_file = os.path.join(self.path, self.lab2idx_file)
        label_file = os.path.join(self.path, self.metainfo_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in self.splits:
                split_path = os.path.join(self.path, split)
                pdb_files += glob.glob(os.path.join(split_path, '*.ent'))
            self.load_pdbs(pdb_files)
            self.save_pickle(pkl_file)

        class_map = json.load(open(map_file, 'r'))
        label_list =  json.load(open(label_file))
        self.class_map = class_map
        self.label_list = label_list

        self.name = [os.path.basename(single)[:-4] for single in self.pdb_files]
        fold_labels = [
            [
                class_map['class'][label_list[single]['lbl']['class']],
                class_map['fold'][label_list[single]['lbl']['fold']],
                class_map['super'][label_list[single]['lbl']['super']]
            ]
            for single in self.name
        ]
        fold_labels = torch.as_tensor(fold_labels)

        if self.multitask:
            self.targets = {
                'class': [s[0] for s in fold_labels], 
                'fold': [s[1] for s in fold_labels], 
                'super':[s[2] for s in fold_labels]
                }
        else:
            self.targets = {
                'super':[s[2] for s in fold_labels]
                }

        self.num_samples = []
        for split in self.splits:
            pdb_files = [single for single in self.pdb_files if split in single]
            self.num_samples.append(len(pdb_files))

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0):
        self.transform = transform
        self.lazy = lazy
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                try:
                    mol = Chem.MolFromPDBFile(pdb_file)
                    if not mol: continue
                    protein = data.Protein.from_molecule(mol)
                    if not protein: continue
                except:
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

    def split(self):
        keys = self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        # print('Len of split is %d'%len(splits))
        return splits

    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        
        if self.multitask:
            item = {"graph": protein, 
                    "super": self.targets["super"][index],
                    "class":self.targets['class'][index],
                    'fold':self.targets['fold'][index]}
        else:
            item = {"graph": protein, 
                    "super": self.targets["super"][index]}
            
        if self.transform:
            item = self.transform(item)
        return item
    
# @R.register("datasets.ScopeHBMultiTask")
# class ScopeHBMultiTask(data.ProteinDataset):
#     # processed_file = "scopehb.pkl.gz"
#     processed_file = "tmp.pkl.gz"
#     splits = ["train", "valid","test"]
#     lab2idx_file = 'lab2idx.json'
#     metainfo_file = 'all.json'

#     def __init__(self, path,radius = 8 ,cutoff=4,verbose=1,hbondmax = 4,**kwargs):
#         path = os.path.expanduser(path)
#         path = os.path.abspath(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path
#         self.residue_name = list('LAGVESKIDTRPNFQYHMCW')
#         self.radius = radius
#         self.cutoff = cutoff
#         self.hbondmax = hbondmax
#         self.hb_type = {k:v for k,v in zip(
#             [*range(-5, -2), *range(3, 6)], range(6)
#         )}

#         pkl_file = os.path.join(self.path, self.processed_file)
#         map_file = os.path.join(self.path, self.lab2idx_file)
#         label_file = os.path.join(self.path, self.metainfo_file)

#         if os.path.exists(pkl_file):
#             self.load_pickle(pkl_file, verbose=verbose, **kwargs)
#         else:
#             pdb_files = []
#             for split in self.splits:
#                 split_path = os.path.join(self.path, split)
#                 pdb_files += glob.glob(os.path.join(split_path, '*.ent'))
#             self.load_pdbs(pdb_files, **kwargs)
#             self.save_pickle(pkl_file)

#         class_map = json.load(open(map_file, 'r'))
#         label_list =  json.load(open(label_file))
#         self.class_map = class_map
#         self.label_list = label_list

#         self.name = [os.path.basename(single)[:-4] for single in self.pdb_files]
#         fold_labels = [
#             [
#                 class_map['class'][label_list[single]['lbl']['class']],
#                 class_map['fold'][label_list[single]['lbl']['fold']],
#                 class_map['super'][label_list[single]['lbl']['super']]
#             ]
#             for single in self.name
#         ]
#         fold_labels = torch.as_tensor(fold_labels)
#         self.targets = {
#             'class': [s[0] for s in fold_labels], 
#             'fold': [s[1] for s in fold_labels], 
#             'super':[s[2] for s in fold_labels]
#             }

#         self.num_samples = []

#         for split in self.splits:
#             pdb_files = [single for single in self.pdb_files if split in single]
#             self.num_samples.append(len(pdb_files))
        
#     def extract_hb_raw(self, pdb_file):
#         dssp = os.path.join(os.path.dirname(os.path.dirname(
#             os.path.abspath(__file__)
#         )), 'dssp')
        
#         dssp_file = pdb_file.replace(".ent", ".dssp")
#         if not os.path.exists(dssp_file):
#             os.system(' '.join([dssp,'-i', pdb_file, '-o', dssp_file]))
#         if not os.path.isfile(dssp_file):
#             return None
#         with open(dssp_file, "r") as f:
#             dssp_data = f.readlines()[28:]
#         data = {"model":{}}
#         for line in dssp_data:
#             idx = int(line[0:5])
#             data["model"][str(idx)] = {
#                     "res": line[13],
#                     "ss": line[16],
#                     "acc": int(line[34:38]),
#                     "nho0p": int(line[39:45]),
#                     "nho0e": float(line[46:50]),
#                     "ohn0p": int(line[50:56]),
#                     "ohn0e": float(line[57:61]),
#                     "nho1p": int(line[61:67]),
#                     "nho1e": float(line[68:72]),
#                     "ohn1p": int(line[72:78]),
#                     "ohn1e": float(line[79:83]),
#                     "phi": float(line[103:109]),
#                     "psi": float(line[109:115]),
#                     "x": float(line[115:122]),
#                     "y": float(line[122:129]),
#                     "z": float(line[129:136]),
#             }
#         json_file = dssp_file.replace(".dssp", ".json")
#         with open(json_file, "w") as f:
#             json.dump(data, f)
#         return data

#     def cal_hbond(self, model_dict):
#         residue_name = self.residue_name
#         radius = self.radius
#         cutoff = self.cutoff
#         hbondmax = self.hbondmax

#         model = model_dict['model']
#         size = len(model)
#         coord = []
#         for _,v in model.items():
#             coord.append([v['x'], v['y'], v['z']])
#         coord = torch.as_tensor(coord)
#         dist2 = torch.cdist(coord.unsqueeze(0), coord.unsqueeze(0)).squeeze().numpy()
#         dist2 = torch.as_tensor(dist2)
#         dist2 = torch.nan_to_num(dist2, nan=torch.inf)

#         # idx/model is in [1, size]
#         # dist in in [0,size)
#         # type of idx is int
#         is_frag = lambda idx: all((i > 0 and i < size) 
#                                     and model[str(i)]['res'] in residue_name
#                                     and dist2[i-1, i] <= cutoff
#                                 for i in range(idx-radius-1, idx+radius+1))
#         edge_list = []
#         for idx, res in model.items():
#             idx = int(idx)
#             if (res['nho0e']) <= hbondmax:
#                 idx1 = idx + (res['nho0p'])
#                 if abs(idx - idx1) > 2 and is_frag(idx) and is_frag(idx1):
#                     if res['nho0p'] in self.hb_type:
#                         hb_type = self.hb_type[res['nho0p']]
#                     else:
#                         hb_type = len(self.hb_type)
#                     edge_list.append([idx-1, idx1-1, hb_type])# idx of graph is in [0,size)
            
#             if (res['nho1e']) <= hbondmax:
#                 idx2 = idx + (res['nho1p'])
#                 if abs(idx - idx2) > 2 and is_frag(idx) and is_frag(idx2):
#                     if res['nho1p'] in self.hb_type:
#                         hb_type = self.hb_type[res['nho1p']]
#                     else:
#                         hb_type = len(self.hb_type)
#                     edge_list.append([idx-1, idx2-1, hb_type])
#         if len(edge_list) < 20:
#             return None
#         edge_list = torch.as_tensor(edge_list)
#         return edge_list

#     def load_pdb(self, pdb_file, verbose = 0):
#         mol = Chem.MolFromPDBFile(pdb_file)
#         if not mol:
#             logger.debug("RDKit cannot read PDB file `%s`" % pdb_file)
#             return None
#         prot = data.Protein.from_molecule(mol)
#         if not prot:
#             logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
#             return None
#         if prot.num_residue < 20:
#             logger.debug("Protein is too short from pdb file `%s`. Ignore this sample." % pdb_file)
#             return None
#         if prot.num_residue > 1022:
#             logger.debug("Protein is too long from pdb file `%s`. Ignore this sample." % pdb_file)
#             return None
        
#         prot = AlphaCarbonNode().forward(prot)
#         hbond_file = str.replace(pdb_file, 'ent', 'json')
#         if not os.path.exists(hbond_file):
#             hbond_raw = self.extract_hb_raw(pdb_file)
#             if hbond_raw is None:
#                 logger.debug("Fail to extract HBond from pdb file `%s`. Ignore this sample." % pdb_file)
#                 return None
#         else:
#             with open(hbond_file, 'r') as fp:
#                 hbond_raw = json.load(fp)

#         # Now Analyze HBonds
#         edge_list = self.cal_hbond(hbond_raw)
#         if edge_list is None:
#             logger.debug("No enough HBond from pdb file `%s`. Ignore this sample." % pdb_file)
#             return None
#         # Finish Analyzing
#         # edge_list = torch.as_tensor([[0, 0, 0]])
#         bond_type = edge_list[:,2]
#         num_relation = torch.as_tensor(len(self.hb_type)+1, device=prot.device)
        
#         protein = data.Protein(edge_list, prot.atom_type, bond_type, num_node=prot.num_atom, num_residue=prot.num_residue,
#                                node_position=prot.node_position, atom_name=prot.atom_name, num_relation = num_relation,
#                                 atom2residue=prot.atom2residue, residue_feature=prot.residue_feature, 
#                                 residue_type=prot.residue_type)
#         return protein
        

    
#     def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0):
#         # num_sample = len(pdb_files)
#         # if num_sample > 1000000:
#         #     warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
#         #                 "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

#         self.transform = transform
#         self.lazy = lazy
#         self.data = []
#         self.pdb_files = []
#         self.sequences = []

#         if verbose:
#             pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
#         for i, pdb_file in enumerate(pdb_files):
#             if not lazy or i == 0:
#                 try:    protein = self.load_pdb(pdb_file)
#                 except: 
#                     continue
#                 if protein is None:
#                     continue
#             else:
#                 protein = None
#             if hasattr(protein, "residue_feature"):
#                 with protein.residue():
#                     protein.residue_feature = protein.residue_feature.to_sparse()
#             self.data.append(protein)
#             self.pdb_files.append(pdb_file)
#             self.sequences.append(protein.to_sequence() if protein else None)

        

#     def split(self):
#         keys = self.splits
#         offset = 0
#         splits = []
#         for split_name, num_sample in zip(self.splits, self.num_samples):
#             if split_name in keys:
#                 split = torch_data.Subset(self, range(offset, offset + num_sample))
#                 splits.append(split)
#             offset += num_sample
#         # print('Len of split is %d'%len(splits))
#         return splits


#     def get_item(self, index):
#         if self.lazy:
#             protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
#         else:
#             try: protein = self.data[index].clone()
#             except: print(index)

#         if hasattr(protein, "residue_feature"):
#             with protein.residue():
#                 protein.residue_feature = protein.residue_feature.to_dense()
#         # Superfamily ONLY
#         item = {"graph": protein, 
#                 "super": self.targets["super"][index],
#                 "class":self.targets['class'][index],
#                 'fold':self.targets['fold'][index]}
#         # item = {"graph": protein, 
#         #         "super": self.targets[index][2],
#         #         "class":self.targets[index][0],
#         #         'fold':self.targets[index][1]}
#         # item = {"graph": protein, 
#         #         "target": [
#         #             self.targets['class'][index],
#         #             self.targets['fold'][index],
#         #             self.targets['super'][index]
#         #       

#         if self.transform:
#             item = self.transform(item)
#         return item

@R.register("datasets.ScopeHB")
class ScopeHB(data.ProteinDataset):
    # processed_file = "scopehb.pkl.gz"
    processed_file = "tmp.pkl.gz"
    splits = ["train", "valid","test"]
    lab2idx_file = 'lab2idx.json'
    metainfo_file = 'all.json'

    def __init__(self, path,radius = 8 ,cutoff=4,verbose=1,hbondmax = 4,multitask=False,**kwargs):
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.residue_name = list('LAGVESKIDTRPNFQYHMCW')
        self.radius = radius
        self.cutoff = cutoff
        self.hbondmax = hbondmax
        self.hb_type = {k:v for k,v in zip(
            [*range(-5, -2), *range(3, 6)], range(6)
        )}
        self.multitask =multitask

        pkl_file = os.path.join(self.path, self.processed_file)
        map_file = os.path.join(self.path, self.lab2idx_file)
        label_file = os.path.join(self.path, self.metainfo_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in self.splits:
                split_path = os.path.join(self.path, split)
                pdb_files += glob.glob(os.path.join(split_path, '*.ent'))
            self.load_pdbs(pdb_files, **kwargs)
            self.save_pickle(pkl_file)

        class_map = json.load(open(map_file, 'r'))
        label_list =  json.load(open(label_file))
        self.class_map = class_map
        self.label_list = label_list


        self.name = [os.path.basename(single)[:-4] for single in self.pdb_files]
        fold_labels = [
            [
                class_map['class'][label_list[single]['lbl']['class']],
                class_map['fold'][label_list[single]['lbl']['fold']],
                class_map['super'][label_list[single]['lbl']['super']]
            ]
            for single in self.name
        ]
        fold_labels = torch.as_tensor(fold_labels)

        if self.multitask:
            self.targets = {
                'class': [s[0] for s in fold_labels], 
                'fold': [s[1] for s in fold_labels], 
                'super':[s[2] for s in fold_labels]
                }
        else:
            self.targets = {
                'super':[s[2] for s in fold_labels]
                }

        self.num_samples = []

        for split in self.splits:
            pdb_files = [single for single in self.pdb_files if split in single]
            self.num_samples.append(len(pdb_files))
        
    def extract_hb_raw(self, pdb_file):
        dssp = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )), 'dssp')
        
        dssp_file = pdb_file.replace(".ent", ".dssp")
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

    def cal_hbond(self, model_dict):
        residue_name = self.residue_name
        radius = self.radius
        cutoff = self.cutoff
        hbondmax = self.hbondmax

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
                    if res['nho0p'] in self.hb_type:
                        hb_type = self.hb_type[res['nho0p']]
                    else:
                        hb_type = len(self.hb_type)
                    edge_list.append([idx-1, idx1-1, hb_type])# idx of graph is in [0,size)
            
            if (res['nho1e']) <= hbondmax:
                idx2 = idx + (res['nho1p'])
                if abs(idx - idx2) > 2 and is_frag(idx) and is_frag(idx2):
                    if res['nho1p'] in self.hb_type:
                        hb_type = self.hb_type[res['nho1p']]
                    else:
                        hb_type = len(self.hb_type)
                    edge_list.append([idx-1, idx2-1, hb_type])
        if len(edge_list) < 20:
            return None
        edge_list = torch.as_tensor(edge_list)
        return edge_list

    def load_pdb(self, pdb_file, verbose = 0):
        mol = Chem.MolFromPDBFile(pdb_file)
        if not mol:
            logger.debug("RDKit cannot read PDB file `%s`" % pdb_file)
            return None
        prot = data.Protein.from_molecule(mol)
        if not prot:
            logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        if prot.num_residue < 20:
            logger.debug("Protein is too short from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        if prot.num_residue > 1022:
            logger.debug("Protein is too long from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        
        prot = AlphaCarbonNode().forward(prot)
        hbond_file = str.replace(pdb_file, 'ent', 'json')
        if not os.path.exists(hbond_file):
            hbond_raw = self.extract_hb_raw(pdb_file)
            if hbond_raw is None:
                logger.debug("Fail to extract HBond from pdb file `%s`. Ignore this sample." % pdb_file)
                return None
        else:
            with open(hbond_file, 'r') as fp:
                hbond_raw = json.load(fp)

        # Now Analyze HBonds
        edge_list = self.cal_hbond(hbond_raw)
        if edge_list is None:
            logger.debug("No enough HBond from pdb file `%s`. Ignore this sample." % pdb_file)
            return None
        # Finish Analyzing
        # edge_list = torch.as_tensor([[0, 0, 0]])
        bond_type = edge_list[:,2]
        num_relation = torch.as_tensor(len(self.hb_type)+1, device=prot.device)
        
        protein = data.Protein(edge_list, prot.atom_type, bond_type, num_node=prot.num_atom, num_residue=prot.num_residue,
                               node_position=prot.node_position, atom_name=prot.atom_name, num_relation = num_relation,
                                atom2residue=prot.atom2residue, residue_feature=prot.residue_feature, 
                                residue_type=prot.residue_type)
        return protein
        

    
    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0):
        # num_sample = len(pdb_files)
        # if num_sample > 1000000:
        #     warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
        #                 "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

        self.transform = transform
        self.lazy = lazy
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                try:    protein = self.load_pdb(pdb_file)
                except: 
                    continue
                if protein is None:
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

        

    def split(self):
        keys = self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        # print('Len of split is %d'%len(splits))
        return splits


    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            try: protein = self.data[index].clone()
            except: print(index)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        if self.multitask:
            item = {"graph": protein, 
                    "super": self.targets["super"][index],
                    "class":self.targets['class'][index],
                    'fold':self.targets['fold'][index]}
        else:
            item = {"graph": protein, 
                    "super": self.targets["super"][index]}

        if self.transform:
            item = self.transform(item)
        return item


    
# @R.register("datasets.Fold3D")
# class Fold3D(data.ProteinDataset):

#     url = "https://zenodo.org/record/7593591/files/fold3d.zip"
#     md5 = "7b052a94afa4c66f9bebeb9efd769186"
#     processed_file = "fold3d.pkl.gz"
#     splits = ["train", "valid", "test_fold", "test_family", "test_superfamily"]

#     def __init__(self, path, test_split="test_fold", verbose=1, **kwargs):
#         path = os.path.expanduser(path)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         self.path = path
#         if test_split not in self.splits[-3:]:
#             raise ValueError("Unknown test split `%s` for Fold3D dataset" % test_split)
#         self.test_split = test_split

#         zip_file = utils.download(self.url, path, md5=self.md5)
#         path = os.path.join(utils.extract(zip_file), "fold3d")
#         pkl_file = os.path.join(path, self.processed_file)

#         if os.path.exists(pkl_file):
#             self.load_pickle(pkl_file, verbose=verbose, **kwargs)
#         else:
#             pdb_files = []
#             for split in self.splits:
#                 split_path = utils.extract(os.path.join(path, "%s.zip" % split))
#                 pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.hdf5")))
#             self.load_hdf5s(pdb_files, verbose=verbose, **kwargs)
#             self.save_pickle(pkl_file, verbose=verbose)

#         label_files = [os.path.join(path, '%s.txt' % split) for split in self.splits]
#         class_map = os.path.join(path, 'class_map.txt')
#         label_list = self.get_label_list(label_files, class_map)
#         fold_labels = [label_list[os.path.basename(pdb_file)[:-5]] for pdb_file in self.pdb_files]
#         self.targets = {'fold_label': fold_labels}

#         splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
#         self.num_samples = [splits.count(split) for split in self.splits]

#     def load_hdf5(self, hdf5_file):
#         h5File = h5py.File(hdf5_file, "r")
#         node_position = torch.as_tensor(h5File["atom_pos"][(0)])
#         num_atom = node_position.shape[0]
#         atom_type = torch.as_tensor(h5File["atom_types"][()])
#         atom_name = h5File["atom_names"][()]
#         atom_name = torch.as_tensor([data.Protein.atom_name2id.get(name.decode(), -1) for name in atom_name])
#         atom2residue = torch.as_tensor(h5File["atom_residue_id"][()])
#         residue_type_name = h5File["atom_residue_names"][()]
#         residue_type = []
#         residue_feature = []
#         lst_residue = -1
#         for i in range(num_atom):
#             if atom2residue[i] != lst_residue:
#                 residue_type.append(data.Protein.residue2id.get(residue_type_name[i].decode(), 0))
#                 residue_feature.append(data.feature.onehot(residue_type_name[i].decode(), data.feature.residue_vocab, allow_unknown=True))
#                 lst_residue = atom2residue[i]
#         residue_type = torch.as_tensor(residue_type)
#         residue_feature = torch.as_tensor(residue_feature)
#         num_residue = residue_type.shape[0]
       
#         '''
#         edge_list = torch.cat([
#             torch.as_tensor(h5File["cov_bond_list"][()]),
#             torch.as_tensor(h5File["cov_bond_list_hb"][()])
#         ], dim=0)
#         bond_type = torch.zeros(edge_list.shape[0], dtype=torch.long)
#         edge_list = torch.cat([edge_list, bond_type.unsqueeze(-1)], dim=-1)
#         '''
#         edge_list = torch.as_tensor([[0, 0, 0]])
#         bond_type = torch.as_tensor([0])

#         protein = data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
#                                node_position=node_position, atom_name=atom_name,
#                                 atom2residue=atom2residue, residue_feature=residue_feature, 
#                                 residue_type=residue_type)
#         return protein

#     def load_hdf5s(self, hdf5_files, transform=None, lazy=False, verbose=0):
#         num_sample = len(hdf5_files)
#         if num_sample > 1000000:
#             warnings.warn("Preprocessing proteins of a large dataset consumes a lot of CPU memory and time. "
#                           "Use load_pdbs(lazy=True) to construct molecules in the dataloader instead.")

#         self.transform = transform
#         self.lazy = lazy
#         self.data = []
#         self.pdb_files = []
#         self.sequences = []

#         if verbose:
#             hdf5_files = tqdm(hdf5_files, "Constructing proteins from pdbs")
#         for i, hdf5_file in enumerate(hdf5_files):
#             if not lazy or i == 0:
#                 protein = self.load_hdf5(hdf5_file)
#             else:
#                 protein = None
#             if hasattr(protein, "residue_feature"):
#                 with protein.residue():
#                     protein.residue_feature = protein.residue_feature.to_sparse()
#             self.data.append(protein)
#             self.pdb_files.append(hdf5_file)
#             self.sequences.append(protein.to_sequence() if protein else None)

#     def get_label_list(self, label_files, classmap):
#         with open(classmap, "r") as fin:
#             lines = [line.strip() for line in fin.readlines()]
#             class_map = dict([line.split('\t') for line in lines])
#         label_list = {}
#         for fname in label_files:
#             label_file = open(fname, 'r')
#             for line in label_file.readlines():
#                 line = line.strip().split('\t')
#                 name, label = line[0], line[-1]
#                 label_list[name] = torch.tensor(int(class_map[label])).long()
#         return label_list

#     def split(self):
#         keys = ["train", "valid", self.test_split]
#         offset = 0
#         splits = []
#         for split_name, num_sample in zip(self.splits, self.num_samples):
#             if split_name in keys:
#                 split = torch_data.Subset(self, range(offset, offset + num_sample))
#                 splits.append(split)
#             offset += num_sample
#         return splits
    
#     def get_item(self, index):
#         if self.lazy:
#             protein = self.load_hdf5(self.pdb_files[index])
#         else:
#             protein = self.data[index].clone()
#         if hasattr(protein, "residue_feature"):
#             with protein.residue():
#                 protein.residue_feature = protein.residue_feature.to_dense()
#         item = {"graph": protein, "fold_label": self.targets["fold_label"][index]}
#         if self.transform:
#             item = self.transform(item)
#         return item
