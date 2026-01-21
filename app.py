import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures, Descriptors, Lipinski, FilterCatalog
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def check_pains(mol):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return "Detected" if catalog.HasMatch(mol) else "Clean"

def calculate_all_data(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())]
    
    # Comprehensive Data Dictionary
    data = {
        "Basic": {
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
        },
        "Quantum/MOPAC": {
            "Mol Refractivity": round(Descriptors.MolMR(mol), 2),
            "Max Partial Charge": round(max(charges), 4),
            "Min Partial Charge": round(min(charges), 4),
            "Labute ASA": round(Descriptors.LabuteASA(mol), 2),
            "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
        },
        "Lead Optimization": {
            "PAINS Alert": check_pains(mol),
            "RO5 Violations": sum([Descriptors.MolWt(mol) > 500, Descriptors.MolLogP(mol) > 5, 
                                 Lipinski.NumHDonors(mol) > 5, Lipinski.NumHAcceptors(mol) > 10]),
            "Veber Rule": "Pass" if (Lipinski.NumRotatableBonds(mol) <= 10 and Descriptors.TPSA(mol) <= 140) else "Fail",
            "Ghose Filter": "Pass" if (160 <= Descriptors.MolWt(mol) <= 480 and -0.4 <= Descriptors.MolLogP(mol) <= 5.6) else "Fail"
        }
    }
    return data

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

main_tab1, main_tab2 = st.tabs(["Single Molecule Analysis", "Batch Screening"])

with main_tab1:
    smiles_input = st.text_input("Enter SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
    num_conf = st.slider("Conformers to Generate", 1, 50, 10)
    
    if smiles_input:
        mol_base = Chem.MolFromSmiles(smiles_input)
        if mol_base:
            all_data = calculate_all_data(mol_base)
            energy_data, mol_ready = generate_conformers(mol_base, num_conf)
            
            # --- Data Display Selection ---
            st.subheader("Analysis Results")
            display_mode = st.selectbox("Select Data to Display:", 
                                      ["Physicochemical Properties", "Quantum & MOPAC Proxies", "Lead Optimization Metrics"])
            
            cols = st.columns(len(all_data["Basic"]))
            if display_mode == "Physicochemical Properties":
                for i, (k, v) in enumerate(all_data["Basic"].items()):
                    cols[i % len(cols)].metric(k, v)
            elif display_mode == "Quantum & MOPAC Proxies":
                q_cols = st.columns(len(all_data["Quantum/MOPAC"]))
                for i, (k, v) in enumerate(all_data["Quantum/MOPAC"].items()):
                    q_cols[i % len(q_cols)].metric(k, v)
            else:
                l_cols = st.columns(len(all_data["Lead Optimization"]))
                for i, (k, v) in enumerate(all_data["Lead Optimization"].items()):
                    l_cols[i % len(l_cols)].metric(k, v)

            if energy_data:
                df = pd.DataFrame(energy_data).sort_values("Stability Score (Rel)", ascending=False)
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("Conformer Stability")
                    st.dataframe(df, use_container_width=True)
                    sel_id = st.selectbox("Select ID for 3D View", df["ID"].tolist())
                    st.download_button("Download PDB", Chem.MolToPDBBlock(mol_ready, confId=int(sel_id)), f"conf_{sel_id}.pdb")

                with col_right:
                    st.subheader(f"3D Visualizer (ID: {sel_id})")
                    view = py3Dmol.view(width=800, height=500)
                    view.addModel(Chem.MolToMolBlock(mol_ready, confId=int(sel_id)), 'mol')
                    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                    view.zoomTo()
                    showmol(view, height=500, width=800)

with main_tab2:
    st.subheader("High-Throughput Batch Screening")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if "SMILES" in batch_df.columns:
            results = []
            for sm in batch_df["SMILES"]:
                m = Chem.MolFromSmiles(str(sm))
                if m:
                    p = calculate_all_data(m)
                    # Flattening dictionary for dataframe
                    flat_p = {**p["Basic"], **p["Quantum/MOPAC"], **p["Lead Optimization"], "SMILES": sm}
                    results.append(flat_p)
            st.dataframe(pd.DataFrame(results), use_container_width=True)
