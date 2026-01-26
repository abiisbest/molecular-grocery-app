import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Quantum Ligand Explorer", layout="wide")

def get_fmo_descriptors(mol):
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    homo = -5.5 - (0.1 * logp) + (0.01 * tpsa)
    lumo = -1.2 + (0.05 * logp) - (0.02 * tpsa)
    gap = lumo - homo
    eta = gap / 2  
    mu = (homo + lumo) / 2  
    omega = (mu**2) / (2 * eta) if eta != 0 else 0 
    return {"HOMO": round(homo, 3), "LUMO": round(lumo, 3), "Gap": round(gap, 3), "Hardness": round(eta, 3), "Electrophilicity": round(omega, 3)}

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=AllChem.ETKDGv3())
    res = []
    for cid in range(mol.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
        if ff:
            ff.Minimize()
            res.append({"ID": cid, "E": ff.CalcEnergy()})
    if not res: return [], mol
    min_e = min(r["E"] for r in res)
    for r in res: r["Rel_E"] = round(r["E"] - min_e, 4)
    return sorted(res, key=lambda x: x["Rel_E"]), mol

st.title("⚛️ Quantum FMO Analyzer")

c_top1, c_top2 = st.columns([3, 1])
smiles = c_top1.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
n_conf = c_top2.number_input("Conformers", 1, 100, 10)

mol = Chem.MolFromSmiles(smiles)

if mol:
    fmo = get_fmo_descriptors(mol)
    conf_data, mol_hs = generate_conformers(mol, n_conf)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("HOMO (eV)", fmo["HOMO"])
    m2.metric("LUMO (eV)", fmo["LUMO"])
    m3.metric("Gap (eV)", fmo["Gap"])
    m4.metric("Hardness (η)", fmo["Hardness"])
    m5.metric("Electrophilicity (ω)", fmo["Electrophilicity"])
    m6.metric("MW", round(Descriptors.MolWt(mol), 1))

    st.divider()

    v1, v2, v3 = st.columns([1.5, 1, 1])

    with v1:
        # Sorted IDs: Most stable (lowest Rel_E) to least stable
        sorted_ids = [r['ID'] for r in conf_data]
        sel_id = st.selectbox("Active Conformer ID (Ranked by Stability)", sorted_ids)
        
        view = py3Dmol.view(width=450, height=400)
        view.addModel(Chem.MolToMolBlock(mol_hs, confId=sel_id), 'mol')
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
        view.zoomTo()
        showmol(view, height=400, width=450)

    with v2:
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['LUMO'], fmo['LUMO']], name="LUMO", line=dict(color='RoyalBlue', width=6)))
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['HOMO'], fmo['HOMO']], name="HOMO", line=dict(color='Crimson', width=6)))
        fig_gap.update_layout(title="FMO Energy Level", yaxis_title="eV", height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gap, use_container_width=True)

    with v3:
        df_pes = pd.DataFrame(conf_data)
        fig_pes = go.Figure(data=go.Scatter(x=list(range(len(df_pes))), y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
        fig_pes.update_layout(title="PES (Stability Trend)", xaxis_title="Stability Rank", yaxis_title="ΔE (kcal/mol)", height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pes, use_container_width=True)

    st.divider()
    
    d1, d2 = st.columns(2)
    with d1:
        st.write("**Frontier Orbital Indices**")
        st.dataframe(pd.DataFrame([fmo]), use_container_width=True, hide_index=True)
    with d2:
        st.write("**Conformer Energy Table (Sorted)**")
        st.dataframe(df_pes, use_container_width=True, hide_index=True)
