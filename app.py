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
    if not mol:
        return None, None
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5 
    
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids:
        return None, None

    data = []
    for conf_id in ids:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff:
            ff.Minimize()
            energy = ff.CalcEnergy()
            data.append({"ID": conf_id, "Raw Energy": energy})
            
    if not data:
        return None, None

    # Calculate Relative Energy (Delta E)
    min_energy = min(d["Raw Energy"] for d in data)
    for d in data:
        # Most stable conformer becomes 0.0
        d["Relative Energy (kcal/mol)"] = round(d["Raw Energy"] - min_energy, 4)
            
    return mol, data

st.title("Molecular Property & Conformational Platform")

smiles_input = st.text_input("Enter 2D SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
num_conf = st.slider("Conformers to Generate", 1, 50, 10)

if smiles_input:
    mol, energy_data = generate_conformers(smiles_input, num_conf)
    
    if mol and energy_data:
        # Show only relevant columns
        df = pd.DataFrame(energy_data)[["ID", "Relative Energy (kcal/mol)"]].sort_values("Relative Energy (kcal/mol)")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Stability Ranking")
            st.write("Energy relative to the most stable conformer (0.00).")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "energies.csv", "text/csv")
            
            st.divider()
            selected_id = st.selectbox("Select ID for 3D View", df["ID"])
            
            pdb_block = Chem.MolToPDBBlock(mol, confId=int(selected_id))
            st.download_button("Download PDB", pdb_block, f"conf_{selected_id}.pdb", "chemical/x-pdb")
            
        with col2:
            st.subheader(f"3D View - Conformer {selected_id}")
            feats = get_pharmacophores(mol)
            view = py3Dmol.view(width=800, height=500)
            mb = Chem.MolToMolBlock(mol, confId=int(selected_id))
            view.addModel(mb, 'mol')
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            
            for f in feats:
                p = f.GetPos()
                fam = f.GetFamily()
                color = "blue" if fam == "Donor" else "red" if fam == "Acceptor" else "orange"
                view.addSphere({'center':{'x':p.x,'y':p.y,'z':p.z}, 'radius':0.7, 'color':color, 'opacity':0.5})
            
            view.zoomTo()
            showmol(view, height=500, width=800)
    else:
        st.error("Check SMILES validity.")
