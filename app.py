import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Molecular Analysis Platform", layout="wide")

def get_pharmacophores(mol):
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    return factory.GetFeaturesForMol(mol)

def generate_conformers(smiles, num_conf):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, num_confs=num_conf, params=AllChem.ETKDG())
    energies = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff:
            energies.append(ff.CalcEnergy())
        else:
            energies.append(float('nan'))
    return mol, energies

st.title("Molecular Property & Conformational Analysis")
smiles_input = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

if smiles_input:
    mol, energies = generate_conformers(smiles_input, 10)
    if mol:
        col1, col2 = st.columns([1, 2])
        df = pd.DataFrame({"ID": range(len(energies)), "Energy": energies}).dropna().sort_values("Energy")
        
        with col1:
            st.dataframe(df)
            selected = st.selectbox("Select Conformer ID", df["ID"])
            
        with col2:
            feats = get_pharmacophores(mol)
            view = py3Dmol.view(width=800, height=500)
            view.addModel(Chem.MolToMolBlock(mol, confId=int(selected)), 'mol')
            view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
            
            for f in feats:
                p = f.GetPos()
                fam = f.GetFamily()
                color = "blue" if fam == "Donor" else "red" if fam == "Acceptor" else "orange"
                view.addSphere({'center':{'x':p.x,'y':p.y,'z':p.z}, 'radius':0.7, 'color':color, 'opacity':0.6})
            
            view.zoomTo()
            showmol(view, height=500, width=800)
