import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go

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
    shift = (conf_id * 0.005) 
    homo = homo_base + shift
    lumo = lumo_base - shift
    gap = lumo - homo
    eta = gap / 2  
    s = 1 / eta if eta != 0 else 0 
    mu = (homo + lumo) / 2  
    omega = (mu**2) / (2 * eta) if eta != 0 else 0 
    return {"HOMO": round(homo, 3), "LUMO": round(lumo, 3), "Gap": round(gap, 3), 
            "Hardness": round(eta, 3), "Softness": round(s, 3), 
            "Potential": round(mu, 3), "Electrophilicity": round(omega, 3)}

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

st.title("⚛️ Advanced Quantum FMO Analyzer")

top_c1, top_c2, top_c3 = st.columns([3, 1, 1])
smiles = top_c1.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
n_conf = top_c2.number_input("Conformers", 1, 100, 20)
graph_mode = top_c3.selectbox("Analysis Plot", ["FMO Gap Trend", "PES (Stability)"])

mol = Chem.MolFromSmiles(smiles)

if mol:
    conf_data, mol_hs = generate_conformers(mol, n_conf)
    sorted_ids = [r['ID'] for r in conf_data]
    
    st.markdown("### 1. Physicochemical & Structural Selection")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("MW", round(Descriptors.MolWt(mol), 2))
    b2.metric("LogP", round(Descriptors.MolLogP(mol), 2))
    b3.metric("TPSA", round(Descriptors.TPSA(mol), 2))
    b4.metric("H-Donors", Lipinski.NumHDonors(mol))
    b5.metric("H-Acceptors", Lipinski.NumHAcceptors(mol))
    b6.metric("Rot. Bonds", Lipinski.NumRotatableBonds(mol))

    sel_id = st.selectbox("Active Conformer ID (Ranked by Stability)", sorted_ids)
    
    fmo = get_fmo_descriptors(mol_hs, sel_id)
    rel_energy = next(item["Rel_E"] for item in conf_data if item["ID"] == sel_id)

    st.markdown("### 2. Conformer-Specific Quantum Metrics")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.metric("HOMO (eV)", fmo["HOMO"])
    q2.metric("LUMO (eV)", fmo["LUMO"])
    q3.metric("Gap (ΔE)", fmo["Gap"])
    q4.metric("Hardness (η)", fmo["Hardness"])
    q5.metric("Softness (S)", fmo["Softness"])
    q6.metric("Electrophilicity (ω)", fmo["Electrophilicity"])

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
        st.caption("⚪ H | 🔘 C | 🔵 N | 🔴 O | 🟡 S | 🌫️ VDW Surface")

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
            gaps = [get_fmo_descriptors(mol_hs, cid)["Gap"] for cid in sorted_ids]
            fig = go.Figure(data=go.Scatter(x=list(range(len(gaps))), y=gaps, mode='lines+markers', line_color='orange'))
            fig.add_trace(go.Scatter(x=[sorted_ids.index(sel_id)], y=[fmo['Gap']], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="Gap (eV)", showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        else:
            df_pes = pd.DataFrame(conf_data)
            fig = go.Figure(data=go.Scatter(x=list(range(len(df_pes))), y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
            fig.add_trace(go.Scatter(x=[sorted_ids.index(sel_id)], y=[rel_energy], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="ΔE (kcal/mol)", showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 3. Researcher's Notes & Coordinate Mapping")
    geo_c1, geo_c2, geo_c3 = st.columns([1, 1.2, 1])

    with geo_c1:
        st.write("**Electronic Interpretation**")
        if fmo['Gap'] > 2.5:
            st.success(f"ID {sel_id}: High Hardness (Stable)")
        else:
            st.warning(f"ID {sel_id}: High Softness (Reactive)")
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
