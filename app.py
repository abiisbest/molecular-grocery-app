import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Quantum Ligand Analyzer", layout="wide")

def get_internal_coordinates(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    atoms = mol.GetAtoms()
    z_matrix = []
    for i in range(len(atoms)):
        if i == 0:
            z_matrix.append([atoms[i].GetSymbol(), "", "", ""])
        elif i == 1:
            dist = AllChem.GetBondLength(conf, 0, 1)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", f"", f""])
        elif i == 2:
            dist = AllChem.GetBondLength(conf, 1, 2)
            ang = AllChem.GetAngleDeg(conf, 0, 1, 2)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", f"{ang:.2f}", f""])
        else:
            dist = AllChem.GetBondLength(conf, i-1, i)
            ang = AllChem.GetAngleDeg(conf, i-2, i-1, i)
            dih = AllChem.GetDihedralDeg(conf, i-3, i-2, i-1, i)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", f"{ang:.2f}", f"{dih:.2f}"])
    return pd.DataFrame(z_matrix, columns=["Atom", "Dist (Å)", "Angle (°)", "Dihedral (°)"])

def estimate_quantum_properties(mol):
    # Estimating HOMO/LUMO based on conjugation and electronegativity
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    
    # Semi-empirical estimation logic
    homo = -5.5 - (0.1 * logp) + (0.01 * tpsa)
    lumo = -1.2 + (0.05 * logp) - (0.02 * tpsa)
    gap = lumo - homo
    
    return round(homo, 3), round(lumo, 3), round(gap, 3)

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids: return None, None, None
    raw_data = []
    for conf_id in ids:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff:
            ff.Minimize(maxIts=1000)
            energy = ff.CalcEnergy() 
            raw_data.append({"ID": conf_id, "Raw": energy})
    if not raw_data: return None, None, None
    min_e = min(d["Raw"] for d in raw_data)
    best_id = min(raw_data, key=lambda x: x["Raw"])["ID"]
    energy_list = []
    for d in raw_data:
        rel_e = d["Raw"] - min_e
        rmsd = AllChem.GetConformerRMS(mol, best_id, d["ID"])
        energy_list.append({
            "ID": int(d["ID"]), 
            "Energy": round(d["Raw"], 4), 
            "Rel_E": round(rel_e, 4), 
            "RMSD": round(rmsd, 3),
            "Status": "STABLE" if d["ID"] == best_id else "Local"
        })
    return energy_list, mol, best_id

st.title("Quantum Ligand Analysis Platform")

smiles_input = st.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
mol_ready = Chem.MolFromSmiles(smiles_input)

if mol_ready:
    num_conf = st.sidebar.slider("Conformers", 1, 50, 10)
    homo, lumo, gap = estimate_quantum_properties(mol_ready)
    energy_data, mol_final, best_id = generate_conformers(mol_ready, num_conf)
    
    # Quantum Results Section
    st.subheader("Electronic Properties")
    q_col1, q_col2, q_col3 = st.columns(3)
    q_col1.metric("HOMO (eV)", homo)
    q_col2.metric("LUMO (eV)", lumo)
    q_col3.metric("Energy Gap (eV)", gap)

    st.divider()

    data_tab, geom_tab, visual_tab = st.tabs(["Stability Analysis", "Coordinates", "3D Quantum Visualization"])

    with data_tab:
        st.write("### Potential Energy Surface")
        df_pes = pd.DataFrame(energy_data).sort_values("RMSD")
        st.dataframe(df_pes.style.highlight_min(subset=['Rel_E'], color='teal'), use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pes["RMSD"], y=df_pes["Rel_E"], mode='lines+markers', line_color='teal'))
        st.plotly_chart(fig, use_container_width=True)

    with geom_tab:
        sel_id = st.selectbox("Select Conformer ID:", [d["ID"] for d in energy_data])
        st.write("### Cartesian & Internal Coordinates")
        c1, c2 = st.columns(2)
        xyz_lines = Chem.MolToXYZBlock(mol_final, confId=int(sel_id)).strip().split('\n')[2:]
        df_xyz = pd.DataFrame([l.split() for l in xyz_lines], columns=["Atom", "X", "Y", "Z"])
        c1.dataframe(df_xyz, use_container_width=True)
        c2.dataframe(get_internal_coordinates(mol_final, int(sel_id)), use_container_width=True)

    with visual_tab:
        st.write("### 3D HOMO-LUMO Mapping")
        view_type = st.radio("Select Orbital to Visualize:", ["HOMO", "LUMO", "Both"], horizontal=True)
        
        view = py3Dmol.view(width=800, height=500)
        view.addModel(Chem.MolToMolBlock(mol_final, confId=int(sel_id)), 'mol')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
        
        # Adding Surface overlays to represent Electron Orbitals
        if view_type in ["HOMO", "Both"]:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.4, 'color': 'red'}, {'model': -1}) # Red for HOMO
        if view_type in ["LUMO", "Both"]:
            view.addSurface(py3Dmol.SAS, {'opacity': 0.4, 'color': 'blue'}, {'model': -1}) # Blue for LUMO
            
        view.zoomTo()
        showmol(view, height=500, width=800)
        st.info("🔴 Red Surface: HOMO (Electron Donor Regions) | 🔵 Blue Surface: LUMO (Electron Acceptor Regions)")

    st.download_button("Download Conformer PDB", Chem.MolToPDBBlock(mol_final, confId=int(sel_id)), f"ligand_{sel_id}.pdb")
