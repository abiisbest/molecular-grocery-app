import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Molecular Platform", layout="wide")

def get_pharmacophores(mol):
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    return factory.GetFeaturesForMol(mol)

def generate_conformers(smiles, num_conf=10):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    
    if not ids:
        return None, None

    data = []
    for conf_id in ids:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff:
            ff.Minimize()
            energy = ff.CalcEnergy()
            data.append({"ID": conf_id, "Energy (kcal/mol)": energy})
            
    return mol, data

st.title("Integrated Computational Platform for Molecular Property Prediction")
st.markdown("### Conformational Energy Analysis & Pharmacophore Mapping")

smiles_input = st.text_input("Enter 2D SMILES:", "c1ccccc1C(=O)O")

if smiles_input:
    num_conf = st.sidebar.slider("Number of Conformers", 1, 50, 10)
    
    mol, energy_data = generate_conformers(smiles_input, num_conf)
    
    if mol and energy_data:
        df = pd.DataFrame(energy_data).sort_values("Energy (kcal/mol)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Conformational Energy")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Energy Data as CSV",
                data=csv,
                file_name="conformational_energies.csv",
                mime="text/csv"
            )
            
            st.divider()
            
            selected_id = st.selectbox("Select Conformer ID for 3D View", df["ID"])
            
            pdb_block = Chem.MolToPDBBlock(mol, confId=int(selected_id))
            st.download_button(
                label="Download Selected Conformer (PDB)",
                data=pdb_block,
                file_name=f"conformer_{selected_id}.pdb",
                mime="chemical/x-pdb"
            )
            
            st.subheader("Pharmacophore Legend")
            st.write("🔵 **Donor**")
            st.write("🔴 **Acceptor**")
            st.write("🟠 **Aromatic / Hydrophobe**")
            
        with col2:
            st.subheader(f"3D Visualization (Conformer {selected_id})")
            
            feats = get_pharmacophores(mol)
            
            view = py3Dmol.view(width=800, height=500)
            mb = Chem.MolToMolBlock(mol, confId=int(selected_id))
            view.addModel(mb, 'mol')
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            
            for f in feats:
                pos = f.GetPos()
                fam = f.GetFamily()
                color = "blue" if fam == "Donor" else "red" if fam == "Acceptor" else "orange"
                view.addSphere({
                    'center': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                    'radius': 0.8,
                    'color': color,
                    'opacity': 0.5
                })
            
            view.zoomTo()
            showmol(view, height=500, width=800)
    else:
        st.error("Error processing SMILES or generating conformers.")
