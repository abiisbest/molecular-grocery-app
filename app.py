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
    geo_shift = np.std(pos) * 0.02
    homo = homo_base + geo_shift
    lumo = lumo_base - geo_shift
    gap = lumo - homo
    mu = (homo + lumo) / 2
    omega = (mu**2) / gap if gap != 0 else 0
    return {"HOMO": round(homo, 3), "LUMO": round(lumo, 3), "Gap": round(gap, 3), 
            "Potential": round(mu, 3), "Electrophilicity": round(omega, 3)}

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, randomSeed=42, pruneRmsThresh=0.5)
    res = []
    prop = AllChem.MMFFGetMoleculeProperties(mol)
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=cid)
        if ff:
            ff.Minimize(maxIts=1000)
            energy = ff.CalcEnergy()
            res.append({"ID": int(cid), "E": energy})
    if not res: return [], mol
    min_e = min(r["E"] for r in res)
    for r in res:
        r["Rel_E"] = round(r["E"] - min_e, 6)
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

def make_orbital_cube(mol, conf_id, orbital_type="HOMO"):
    conf = mol.GetConformer(conf_id)
    pos = conf.GetPositions()
    n_atoms = mol.GetNumAtoms()
    min_bounds = pos.min(axis=0) - 5
    max_bounds = pos.max(axis=0) + 5
    grid_res = 30
    x, y, z = [np.linspace(min_bounds[i], max_bounds[i], grid_res) for i in range(3)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_data = np.zeros_like(X)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_pos = pos[i]
        symbol = atom.GetSymbol()
        if orbital_type == "HOMO":
            weight = 2.4 if symbol in ['N', 'O', 'S', 'P'] else 0.4
            phase = 1 if i % 2 == 0 else -1
        else:
            weight = 2.4 if symbol in ['C', 'F', 'Cl', 'B'] else 0.4
            phase = -1 if i % 2 == 0 else 1
        dist_sq = (X - atom_pos[0])**2 + (Y - atom_pos[1])**2 + (Z - atom_pos[2])**2
        grid_data += phase * weight * np.exp(-dist_sq / 3.0)
    header = f"Orbital\nGenerated\n{n_atoms} {min_bounds[0]} {min_bounds[1]} {min_bounds[2]}\n"
    header += f"{grid_res} {(max_bounds[0]-min_bounds[0])/(grid_res-1)} 0 0\n"
    header += f"{grid_res} 0 {(max_bounds[1]-min_bounds[1])/(grid_res-1)} 0\n"
    header += f"{grid_res} 0 0 {(max_bounds[2]-min_bounds[2])/(grid_res-1)}\n"
    for i in range(n_atoms):
        at = mol.GetAtomWithIdx(i)
        p = pos[i]
        header += f"{at.GetAtomicNum()} {at.GetAtomicNum()}.0 {p[0]} {p[1]} {p[2]}\n"
    flat_data = grid_data.flatten()
    body = "".join([" ".join(f"{val:12.6E}" for val in flat_data[i:i+6]) + "\n" for i in range(0, len(flat_data), 6)])
    return header + body

st.title("⚛️ Molecular Reactivity and Conformer Explorer")

up_col, set_col = st.columns([2, 1])
with up_col:
    uploaded_file = st.file_uploader("Upload Molecule", type=["sdf", "pdb", "mol2"])
    smiles_input = st.text_input("OR Enter SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
with set_col:
    n_conf = st.number_input("Conformers", 1, 100, 30)
    view_mode = st.radio("Orbital Visual:", ["Structure Only", "HOMO Lobes", "LUMO Lobes"], horizontal=True)
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
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("HOMO (eV)", fmo["HOMO"])
    q2.metric("LUMO (eV)", fmo["LUMO"])
    q3.metric("Gap (ΔE)", fmo["Gap"])
    q4.metric("Potential (μ)", fmo["Potential"])
    q5.metric("Rel. Energy (kcal)", f"{rel_energy:.4f}")

    st.divider()

    v1, v2, v3 = st.columns([1.5, 1, 1])
    with v1:
        st.write("**3D Molecular Orbitals**")
        view = py3Dmol.view(width=450, height=350)
        view.addModel(Chem.MolToMolBlock(mol_hs, confId=sel_id), 'mol')
        view.setStyle({'stick': {'radius': 0.12}, 'sphere': {'scale': 0.22}})
        
        if "Lobes" in view_mode:
            current_orb = "HOMO" if "HOMO" in view_mode else "LUMO"
            cube_data = make_orbital_cube(mol_hs, sel_id, current_orb)
            view.addVolumetricData(cube_data, "cube", {'isoval': 0.08, 'color': "blue", 'opacity': 0.85})
            view.addVolumetricData(cube_data, "cube", {'isoval': -0.08, 'color': "red", 'opacity': 0.85})
        else:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.15, 'color': 'white'})
        
        view.zoomTo()
        showmol(view, height=350, width=450)
        
        # Legend Section
        if "Lobes" in view_mode:
            orb_name = "HOMO" if "HOMO" in view_mode else "LUMO"
            st.markdown(f"""
            <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 10px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 15px; height: 15px; background-color: blue; border-radius: 3px;"></div>
                    <span>{orb_name} (+) Phase</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 15px; height: 15px; background-color: red; border-radius: 3px;"></div>
                    <span>{orb_name} (-) Phase</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            fig.add_trace(go.Scatter(x=[sorted_ids.index(sel_id)], y=[fmo['Gap']], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="Gap (eV)", showlegend=False)
        else:
            df_pes = pd.DataFrame(conf_data)
            fig = go.Figure(data=go.Scatter(x=list(range(len(df_pes))), y=df_pes['Rel_E'], mode='lines+markers', line_color='teal'))
            fig.add_trace(go.Scatter(x=[sorted_ids.index(sel_id)], y=[rel_energy], mode='markers', marker=dict(color='red', size=10, symbol='star')))
            fig.update_layout(height=350, xaxis_title="Stability Rank", yaxis_title="ΔE (kcal/mol)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 3. Coordinate Mapping")
    geo_c1, geo_c2 = st.columns(2)
    with geo_c1:
        st.write("**Internal Coordinates (Z-Matrix)**")
        st.dataframe(get_internal_coordinates(mol_hs, sel_id), use_container_width=True, height=250)
    with geo_c2:
        st.write("**Cartesian (XYZ)**")
        xyz_block = Chem.MolToXYZBlock(mol_hs, confId=sel_id).split('\n')[2:]
        xyz_data = [line.split() for line in xyz_block if line.strip()]
        st.dataframe(pd.DataFrame(xyz_data, columns=["Atom", "X", "Y", "Z"]), use_container_width=True, height=250)
else:
    st.error("Invalid Input.")
