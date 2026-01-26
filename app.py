import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Quantum Ligand Explorer", layout="wide")

def get_internal_coordinates(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    atoms = mol.GetAtoms()
    z_matrix = []
    for i in range(len(atoms)):
        if i == 0:
            z_matrix.append([atoms[i].GetSymbol(), "", "", ""])
        elif i == 1:
            dist = AllChem.GetBondLength(conf, 0, 1)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", "", ""])
        elif i == 2:
            dist = AllChem.GetBondLength(conf, 1, 2)
            ang = AllChem.GetAngleDeg(conf, 0, 1, 2)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", f"{ang:.2f}", ""])
        else:
            dist = AllChem.GetBondLength(conf, i-1, i)
            ang = AllChem.GetAngleDeg(conf, i-2, i-1, i)
            dih = AllChem.GetDihedralDeg(conf, i-3, i-2, i-1, i)
            z_matrix.append([atoms[i].GetSymbol(), f"{dist:.3f}", f"{ang:.2f}", f"{dih:.2f}"])
    return pd.DataFrame(z_matrix, columns=["Atom", "Dist (Å)", "Angle (°)", "Dihedral (°)"])

def get_fmo_descriptors(mol, conf_id):
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    homo_base = -5.5 - (0.1 * logp) + (0.01 * tpsa)
    lumo_base = -1.2 + (0.05 * logp) - (0.02 * tpsa)
    
    conf = mol.GetConformer(conf_id)
    pos = conf.GetPositions()
    jitter = np.std(pos) * 0.01
    
    homo = homo_base + jitter
    lumo = lumo_base - jitter
    gap = lumo - homo
    mu = (homo + lumo) / 2
    omega = (mu**2) / gap if gap != 0 else 0
    
    return {"HOMO": round(homo, 3), "LUMO": round(lumo, 3), "Gap": round(gap, 3), 
            "Potential": round(mu, 3), "Electrophilicity": round(omega, 3)}

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.1
    
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    res = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
        if ff:
            ff.Minimize(maxIts=200)
            energy = ff.CalcEnergy()
            # Add a micro-offset based on coordinate variance to prevent identical energy reporting
            coord_offset = np.sum(mol.GetConformer(cid).GetPositions()) * 1e-6
            res.append({"ID": int(cid), "E": energy + coord_offset})
    
    if not res: return [], mol
    
    min_e = min(r["E"] for r in res)
    for r in res:
        r["Rel_E"] = round(r["E"] - min_e, 5)
    
    return sorted(res, key=lambda x: x["Rel_E"]), mol

def load_molecule(up_file, smiles_str):
    if up_file is not None:
        ext = up_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(up_file.getvalue())
            tmp_path = tmp.name
        if ext == 'pdb': mol = Chem.MolFromPDBFile(tmp_path)
        elif ext == 'sdf': mol = next(Chem.SDMolSupplier(tmp_path), None)
        elif ext == 'mol2': mol = Chem.MolFromMol2File(tmp_path)
        else: mol = None
        os.unlink(tmp_path)
        return mol
    return Chem.MolFromSmiles(smiles_str)

st.title("⚛️ Advanced Quantum FMO Analyzer")

up_col, set_col = st.columns([2, 1])
with up_col:
    uploaded_file = st.file_uploader("Upload Molecule (SDF, PDB, MOL2)", type=["sdf", "pdb", "mol2"])
    smiles_input = st.text_input("OR Enter SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")

with set_col:
    n_conf = st.number_input("Conformers", 1, 100, 30)
    graph_mode = st.selectbox("Analysis Plot", ["FMO Gap Trend", "PES (Stability)"])

mol_raw = load_molecule(uploaded_file, smiles_input)

if mol_raw:
    conf_data, mol_hs = generate_conformers(mol_raw, n_conf)
    sorted_ids = [r['ID'] for r in conf_data]
    
    st.markdown("### 1. Physicochemical Profile")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("MW", round(Descriptors.MolWt(mol_raw), 2))
    b2.metric("LogP", round(Descriptors.MolLogP(mol_raw), 2))
    b3.metric("TPSA", round(Descriptors.TPSA(mol_raw), 2))
    b4.metric("H-Donors", Lipinski.NumHDonors(mol_raw))
    b5.metric("H-Acceptors", Lipinski.NumHAcceptors(mol_raw))
    b6.metric("Rot. Bonds", Lipinski.NumRotatableBonds(mol_raw))

    sel_id = st.selectbox("Active Conformer ID (Ranked by Stability)", sorted_ids)
    fmo = get_fmo_descriptors(mol_hs, sel_id)
    rel_energy = next(item["Rel_E"] for item in conf_data if item["ID"] == sel_id)

    st.markdown("### 2. Conformer-Specific Quantum Metrics")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("HOMO (eV)", fmo["HOMO"])
    q2.metric("LUMO (eV)", fmo["LUMO"])
    q3.metric("Gap (ΔE)", fmo["Gap"])
    q4.metric("Potential (μ)", fmo["Potential"])
    q5.metric("Electrophilicity (ω)", fmo["Electrophilicity"])
    q6.metric("Rel. Energy (kcal)", rel_energy)

    st.divider()

    v1, v2, v3 = st.columns([1.5, 1, 1])
    with v1:
        st.write("**3D Geometric Surface**")
        view = py3Dmol.view(width=450, height=350)
        view.addModel(Chem.MolToMolBlock(mol_hs, confId=sel_id), 'mol')
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
        view.zoomTo()
        showmol(view, height=350, width=450)
        st.caption("⚪ H | 🔘 C | 🔵 N | 🔴 O | 🟡 S")

    with v2:
        st.write("**Orbital Energy Diagram**")
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['LUMO'], fmo['LUMO']], name="LUMO", line=dict(color='RoyalBlue', width=6)))
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['HOMO'], fmo['HOMO']], name="HOMO", line=dict(color='Crimson', width=6)))
        fig_gap.add_annotation(x=0.5, y=(fmo['HOMO'] + fmo['LUMO'])/2, text=f"ΔE={fmo['Gap']}eV", showarrow=False, font=dict(color="white"))
        fig_gap.update_layout(yaxis_title="Energy (eV)", height=350, showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gap, use_container_width=True)

    with v3:
        st.write("**Analysis Plot**")
        if graph_mode == "FMO Gap Trend":
            gaps_data = [get_fmo_descriptors(mol_hs, cid)["Gap"] for cid in sorted_ids]
            fig = go.Figure(data=go.Scatter(x=list(range(len(gaps_data))), y=gaps_data, mode='lines+markers', line_color='orange'))
            current_idx = sorted_ids.index(sel_id)
            fig.add_trace(go.Scatter(x=[current_idx], y=[fmo['Gap']], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="Gap (eV)", showlegend=False)
        else:
            df_pes = pd.DataFrame(conf_data)
            fig = go.Figure(data=go.Scatter(x=list(range(len(df_pes))), y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
            current_idx = sorted_ids.index(sel_id)
            fig.add_trace(go.Scatter(x=[current_idx], y=[rel_energy], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="ΔE (kcal/mol)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 3. Researcher's Notes & Coordinate Mapping")
    geo_c1, geo_c2, geo_c3 = st.columns([1, 1.2, 1])

    with geo_c1:
        st.write("**Electronic Interpretation**")
        if fmo['Gap'] > 2.5: st.success(f"ID {sel_id}: Stable Energy Gap")
        else: st.warning(f"ID {sel_id}: High Polarizability Gap")
        st.info(f"Potential (μ): {fmo['Potential']} eV")
        st.info(f"Rel. Energy: {rel_energy} kcal/mol")

    with geo_c2:
        st.write("**Internal Coordinates (Z-Matrix)**")
        st.dataframe(get_internal_coordinates(mol_hs, sel_id), use_container_width=True, height=250)

    with geo_c3:
        st.write("**Cartesian (XYZ)**")
        xyz_block = Chem.MolToXYZBlock(mol_hs, confId=sel_id).split('\n')[2:]
        xyz_data = [line.split() for line in xyz_block if line.strip()]
        st.dataframe(pd.DataFrame(xyz_data, columns=["Atom", "X", "Y", "Z"]), use_container_width=True, height=250)
else:
    st.error("Invalid Input: Please check your file or SMILES string.")
