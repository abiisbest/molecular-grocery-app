import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Molecular Analysis Platform", layout="wide")

def get_pharmacophores(mol, conf_id):
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    return factory.GetFeaturesForMol(mol, confId=int(conf_id))

def generate_conformers(smiles, num_conf):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None
    
    mol = Chem.AddHs(mol)
    
    # Calculate Gasteiger Charges to stabilize electrostatics and lower energy
    AllChem.ComputeGasteigerCharges(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5 
    
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids:
        return None, None

    data = []
    for conf_id in ids:
        # MMFF94 provides better localized energy for drug-like molecules
        prop = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=conf_id)
        if ff:
            ff.Minimize(maxIts=500) # Increased iterations for better convergence
            energy = ff.CalcEnergy()
            data.append({"ID": conf_id, "Energy (kcal/mol)": round(energy, 4)})
            
    return mol, data

st.title("Integrated Computational Platform for Molecular Property Prediction")
st.markdown("### Conformational Energy Analysis & Pharmacophore Mapping")

smiles_input = st.text_input("Enter 2D SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")

num_conf = st.slider("Select Number of Conformers to Generate", min_value=1, max_value=50, value=10)

if smiles_input:
    mol, energy_data = generate_conformers(smiles_input, num_conf)
    
    if mol and energy_data:
        df = pd.DataFrame(energy_data).sort_values("Energy (kcal/mol)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Results for {len(df)} Unique Conformers")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Energy Data (CSV)", csv, "energies.csv", "text/csv")
            
            st.divider()
            selected_id = st.selectbox("Select ID for 3D View", df["ID"])
            
            pdb_block = Chem.MolToPDBBlock(mol, confId=int(selected_id))
            st.download_button("Download Selected Conformer (PDB)", pdb_block, f"conf_{selected_id}.pdb", "chemical/x-pdb")
            
            st.subheader("Pharmacophore Legend")
            st.write("🔵 **Donor**")
            st.write("🔴 **Acceptor**")
            st.write("🟠 **Aromatic / Hydrophobe**")
            
        with col2:
            st.subheader(f"3D Visualization (Conformer {selected_id})")
            
            feats = get_pharmacophores(mol, selected_id)
            view = py3Dmol.view(width=800, height=500)
            mb = Chem.MolToMolBlock(mol, confId=int(selected_id))
            view.addModel(mb, 'mol')
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            
            for f in feats:
                p = f.GetPos(int(selected_id)) 
                fam = f.GetFamily()
                color = "blue" if fam == "Donor" else "red" if fam == "Acceptor" else "orange"
                view.addSphere({
                    'center': {'x': p.x, 'y': p.y, 'z': p.z},
                    'radius': 0.8,
                    'color': color,
                    'opacity': 0.5
                })
            
            view.zoomTo()
            showmol(view, height=500, width=800)
    else:
        st.error("Error processing SMILES. Please check structure validity.")
