import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures, Descriptors, Lipinski, FilterCatalog
import py3Dmol
from stmol import showmol
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def calculate_cns_mpo(props):
    # Simplified CNS-MPO Score
    def f(val, low, high):
        if val <= low: return 1.0
        if val >= high: return 0.0
        return (high - val) / (high - low)
    
    # Components for CNS-MPO
    s_logp = f(props["LogP"], 3, 5)
    s_mw = f(props["MW"], 360, 500)
    s_tpsa = f(props["TPSA"], 40, 90) if props["TPSA"] > 40 else f(20 - props["TPSA"], 0, 20)
    s_hbd = f(props["HBD"], 0, 3)
    
    return round(s_logp + s_mw + s_tpsa + s_hbd, 2)

def calculate_complex_properties(mol):
    AllChem.ComputeGasteigerCharges(mol)
    props = {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
        "MolRefractivity": round(Descriptors.MolMR(mol), 2),
    }
    
    # Custom Indices
    props["CNS-MPO"] = calculate_cns_mpo(props)
    props["Drug-Likeness"] = "High" if (props["MW"] < 500 and props["LogP"] < 5) else "Low"
    
    return props

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5 
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids: return None, None
    
    raw_data = []
    for conf_id in ids:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff:
            ff.Minimize(maxIts=1000)
            energy = ff.CalcEnergy() 
            raw_data.append({"ID": conf_id, "Raw": energy})
    
    if not raw_data: return None, None
    
    min_e = min(d["Raw"] for d in raw_data)
    best_id = min(raw_data, key=lambda x: x["Raw"])["ID"]
    
    energy_list = []
    for d in raw_data:
        rel_e = d["Raw"] - min_e
        rmsd = AllChem.GetConformerRMS(mol, best_id, d["ID"])
        energy_list.append({
            "ID": int(d["ID"]), 
            "Energy (kcal/mol)": round(d["Raw"], 4),
            "Stability Score (Rel)": round(-rel_e, 4), 
            "RMSD (Å)": round(rmsd, 3)
        })
    return energy_list, mol

st.title("Integrated Computational Platform for Molecular Property Prediction")

tab1, tab2 = st.tabs(["Single Molecule Analysis", "Batch Screening"])

with tab1:
    smiles_input = st.text_input("Enter SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
    num_conf = st.slider("Conformers", 1, 50, 10)
    
    if smiles_input:
        mol_base = Chem.MolFromSmiles(smiles_input)
        if mol_base:
            props = calculate_complex_properties(mol_base)
            energy_data, mol_ready = generate_conformers(mol_base, num_conf)
            
            st.subheader("Advanced Bio-Descriptors")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CNS-MPO Score", props["CNS-MPO"])
            c2.metric("Molar Refractivity", props["MolRefractivity"])
            c3.metric("Fsp3 (Complexity)", props["Fsp3"])
            c4.metric("Drug-Likeness", props["Drug-Likeness"])
            
            if energy_data:
                df = pd.DataFrame(energy_data).sort_values("Stability Score (Rel)", ascending=False)
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("Conformer Ranking")
                    st.dataframe(df[["ID", "Stability Score (Rel)", "RMSD (Å)"]], use_container_width=True)
                    sel_id = st.selectbox("Select ID for 3D View", df["ID"].tolist())
                    st.download_button("Download PDB", Chem.MolToPDBBlock(mol_ready, confId=int(sel_id)), f"conf_{sel_id}.pdb")

                with col_right:
                    st.subheader(f"3D Visualizer (ID: {sel_id})")
                    view = py3Dmol.view(width=800, height=500)
                    view.addModel(Chem.MolToMolBlock(mol_ready, confId=int(sel_id)), 'mol')
                    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                    view.zoomTo()
                    showmol(view, height=500, width=800)

with tab2:
    st.subheader("High-Throughput Batch Screening")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if "SMILES" in batch_df.columns:
            results = []
            for sm in batch_df["SMILES"]:
                m = Chem.MolFromSmiles(str(sm))
                if m:
                    p = calculate_complex_properties(m)
                    p["SMILES"] = sm
                    results.append(p)
            st.dataframe(pd.DataFrame(results), use_container_width=True)
