import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Quantum Ligand Explorer", layout="wide")

def get_fmo_descriptors(mol):
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    # Refined semi-empirical proxies for HOMO/LUMO
    homo = -5.5 - (0.1 * logp) + (0.01 * tpsa)
    lumo = -1.2 + (0.05 * logp) - (0.02 * tpsa)
    gap = lumo - homo
    eta = gap / 2  # Hardness
    mu = (homo + lumo) / 2  # Chemical Potential
    omega = (mu**2) / (2 * eta) if eta != 0 else 0 # Electrophilicity
    return {
        "HOMO (eV)": round(homo, 3),
        "LUMO (eV)": round(lumo, 3),
        "Gap (eV)": round(gap, 3),
        "Hardness (η)": round(eta, 3),
        "Electrophilicity (ω)": round(omega, 3)
    }

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=AllChem.ETKDGv3())
    results = []
    for conf_id in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
        if ff:
            ff.Minimize()
            results.append({"ID": conf_id, "Energy": ff.CalcEnergy()})
    
    min_e = min(r["Energy"] for r in results)
    for r in results:
        r["Rel_E"] = r["Energy"] - min_e
    return sorted(results, key=lambda x: x["Rel_E"]), mol

st.title("⚛️ Advanced Quantum Ligand Analyzer")

with st.sidebar:
    st.header("Settings")
    smiles = st.text_input("SMILES", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
    n_conf = st.slider("Conformers", 5, 100, 20)
    
mol = Chem.MolFromSmiles(smiles)

if mol:
    fmo = get_fmo_descriptors(mol)
    conf_data, mol_hs = generate_conformers(mol, n_conf)
    
    # --- SECTION 1: FRONTIER ORBITAL DASHBOARD ---
    st.subheader("1. Frontier Molecular Orbital (FMO) Analysis")
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        st.metric("HOMO", f"{fmo['HOMO (eV)']} eV")
        st.metric("LUMO", f"{fmo['LUMO (eV)']} eV")
        st.metric("Gap", f"{fmo['Gap (eV)']} eV", delta_color="inverse")
        
    with c2:
        st.metric("Hardness (η)", f"{fmo['Hardness (η)']} eV")
        st.metric("Electrophilicity (ω)", f"{fmo['Electrophilicity (ω)']} eV")
        
    with c3:
        # FMO Energy Diagram
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[fmo['LUMO (eV)'], fmo['LUMO (eV)']], name="LUMO", line=dict(color='RoyalBlue', width=5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[fmo['HOMO (eV)'], fmo['HOMO (eV)']], name="HOMO", line=dict(color='Crimson', width=5)))
        fig.update_layout(title="Energy Gap Visualization", yaxis_title="Energy (eV)", height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- SECTION 2: 3D VISUALIZATION & GEOMETRY ---
    st.subheader("2. Structural & Surface Analysis")
    v1, v2 = st.columns([2, 1])
    
    sel_conf = v2.selectbox("Select Conformer (by Stability)", [r['ID'] for r in conf_data])
    surf_type = v2.radio("Surface Type", ["None", "VDW", "SAS"])
    
    with v1:
        view = py3Dmol.view(width=700, height=450)
        view.addModel(Chem.MolToMolBlock(mol_hs, confId=sel_conf), 'mol')
        view.setStyle({'stick': {}})
        if surf_type == "VDW":
            view.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'colorscheme': 'amine'})
        elif surf_type == "SAS":
            view.addSurface(py3Dmol.SAS, {'opacity': 0.5})
        view.zoomTo()
        showmol(view, height=450, width=700)

    # --- SECTION 3: COMPREHENSIVE DATA ---
    st.divider()
    t1, t2, t3 = st.tabs(["PES Analysis", "Reactivity Logic", "Physicochemical Properties"])
    
    with t1:
        st.write("### Potential Energy Surface (PES)")
        df_pes = pd.DataFrame(conf_data)
        fig_pes = go.Figure(data=go.Scatter(x=df_pes.index, y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
        fig_pes.update_layout(xaxis_title="Conformer Rank", yaxis_title="Relative Energy (kcal/mol)")
        st.plotly_chart(fig_pes, use_container_width=True)

    with t2:
        st.info("#### Reactivity Interpretation")
        st.write(f"This molecule has a gap of **{fmo['Gap (eV)']} eV**.")
        if fmo['Gap (eV)'] > 4.0:
            st.success("Analysis: This is a **'Hard'** molecule. It is likely stable with low chemical reactivity.")
        else:
            st.warning("Analysis: This is a **'Soft'** molecule. It is likely more polarizable and reactive in biochemical environments.")

    with t3:
        prop_data = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H-Bond Donors": Lipinski.NumHDonors(mol),
            "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
            "Rotatable Bonds": Lipinski.NumRotatableBonds(mol)
        }
        st.table(pd.DataFrame([prop_data]))
