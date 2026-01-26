import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="FMO Frontier Analyzer", layout="wide")

def calculate_reactivity_indices(homo, lumo):
    chi = -(homo + lumo) / 2  
    eta = (lumo - homo) / 2   
    omega = (chi**2) / (2 * eta) if eta != 0 else 0
    return {
        "Electronegativity (χ)": round(chi, 3),
        "Global Hardness (η)": round(eta, 3),
        "Electrophilicity (ω)": round(omega, 3)
    }

def estimate_fmo_properties(mol):
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    
    homo = -5.5 - (0.1 * logp) + (0.01 * tpsa) - (0.001 * mw)
    lumo = -1.2 + (0.05 * logp) - (0.02 * tpsa) + (0.002 * mw)
    gap = lumo - homo
    return homo, lumo, gap

st.title("⚛️ FMO Frontier & Electronic Reactivity platform")
st.markdown("Focusing on **Frontier Molecular Orbital (FMO)** theory for ligand binding affinity.")

smiles_input = st.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
mol = Chem.MolFromSmiles(smiles_input)

if mol:
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    homo, lumo, gap = estimate_fmo_properties(mol)
    indices = calculate_reactivity_indices(homo, lumo)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Frontier Energy Levels")
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(x=[0, 1], y=[lumo, lumo], mode="lines", name="LUMO", line=dict(color="royalblue", width=4)))
        fig_energy.add_trace(go.Scatter(x=[0, 1], y=[homo, homo], mode="lines", name="HOMO", line=dict(color="firebrick", width=4)))
        
        fig_energy.update_layout(
            title=f"Energy Gap: {gap:.3f} eV",
            yaxis_title="Energy (eV)",
            xaxis={'visible': False},
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_energy, use_container_width=True)

    with col2:
        st.subheader("Global Reactivity Indices")
        i_col1, i_col2, i_col3 = st.columns(3)
        i_col1.metric("Electronegativity (χ)", indices["Electronegativity (χ)"])
        i_col2.metric("Hardness (η)", indices["Global Hardness (η)"])
        i_col3.metric("Electrophilicity (ω)", indices["Electrophilicity (ω)"])
        
        with st.expander("Interpret Results"):
            st.write(f"**Chemical Hardness ({indices['Global Hardness (η)']} eV):** " + 
                     ("High stability, less reactive." if indices["Global Hardness (η)"] > 2 else "High reactivity, soft molecule."))

    st.divider()

    vis_col, desc_col = st.columns([2, 1])

    with vis_col:
        st.subheader("Orbital Localization Proxy")
        view = py3Dmol.view(width=700, height=500)
        mol_block = Chem.MolToMolBlock(mol)
        view.addModel(mol_block, 'mol')
        view.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
        
        # Simulate an Isosurface representing the electron density areas
        view.addSurface(py3Dmol.VDW, {'opacity': 0.4, 'color': 'white'})
        
        view.zoomTo()
        showmol(view, height=500, width=700)

    with desc_col:
        st.info("### Orbital Focus")
        st.markdown(f"""
        - **HOMO**: {-homo:.2f} eV (Ionization Potential)
        - **LUMO**: {-lumo:.2f} eV (Electron Affinity)
        
        **Nucleophilic Attack Site:** Likely areas with high TPSA.
        
        **Electrophilic Attack Site:** Likely areas with high Hydrophobicity.
        """)
        
        if st.button("Generate Detailed Electronic Report"):
            report = pd.DataFrame({
                "Parameter": ["HOMO", "LUMO", "Gap", "Hardness", "Softness", "Chemical Potential"],
                "Value": [homo, lumo, gap, indices["Global Hardness (η)"], 1/indices["Global Hardness (η)"], -indices["Electronegativity (χ)"]]
            })
            st.table(report)
            
