import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures, Descriptors, Lipinski, FilterCatalog
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def get_internal_coordinates(mol, conf_id):
    # Generating a Z-Matrix style internal coordinate representation
    conf = mol.GetConformer(conf_id)
    atoms = mol.GetAtoms()
    z_matrix = []
    for i in range(len(atoms)):
        pos = conf.GetAtomPosition(i)
        if i == 0:
            z_matrix.append([atoms[i].GetSymbol(), "", "", ""])
        elif i == 1:
            dist = AllChem.GetBondLength(conf, 0, 1)
            z_matrix.append([atoms[i].GetSymbol(), f"R(1,0): {dist:.3f}", "", ""])
        elif i == 2:
            dist = AllChem.GetBondLength(conf, 1, 2)
            ang = AllChem.GetAngleDeg(conf, 0, 1, 2)
            z_matrix.append([atoms[i].GetSymbol(), f"R(2,1): {dist:.3f}", f"A(2,1,0): {ang:.2f}", ""])
        else:
            dist = AllChem.GetBondLength(conf, i-1, i)
            ang = AllChem.GetAngleDeg(conf, i-2, i-1, i)
            dih = AllChem.GetDihedralDeg(conf, i-3, i-2, i-1, i)
            z_matrix.append([atoms[i].GetSymbol(), f"R: {dist:.3f}", f"A: {ang:.2f}", f"D: {dih:.2f}"])
    return pd.DataFrame(z_matrix, columns=["Atom", "Distance (Å)", "Angle (°)", "Dihedral (°)"])

def calculate_all_data(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())]
    data = {
        "Physicochemical": {
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
        },
        "Quantum & MOPAC": {
            "Mol Refractivity": round(Descriptors.MolMR(mol), 2),
            "Max Partial Charge": round(max(charges), 4),
            "Min Partial Charge": round(min(charges), 4),
            "Labute ASA": round(Descriptors.LabuteASA(mol), 2),
            "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
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
        energy_list.append({"ID": int(d["ID"]), "Energy": round(d["Raw"], 4), "Rel_E": round(rel_e, 4), "RMSD": round(rmsd, 3)})
    return energy_list, mol

st.title("Integrated Computational Platform for Molecular Property Prediction")

input_tab, batch_tab = st.tabs(["Analysis & PES Scan", "Batch Screening"])

with input_tab:
    smiles_input = st.text_input("Enter SMILES string:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
    
    if smiles_input:
        mol_base = Chem.MolFromSmiles(smiles_input)
        if mol_base:
            num_conf = st.sidebar.slider("Conformers", 1, 50, 10)
            all_data = calculate_all_data(mol_base)
            energy_data, mol_ready = generate_conformers(mol_base, num_conf)
            
            # --- PES Graph Section ---
            st.subheader("Potential Energy Surface (PES) Scan")
            df_pes = pd.DataFrame(energy_data).sort_values("RMSD")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pes["RMSD"], y=df_pes["Rel_E"], mode='lines+markers', name='PES Scan'))
            fig.update_layout(xaxis_title="RMSD from Global Minimum (Å)", yaxis_title="Relative Energy (kcal/mol)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            

            st.divider()
            
            # --- Coordinates & 3D Section ---
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("Geometry Analysis")
                coord_mode = st.radio("Coordinate Type:", ["Cartesian (XYZ)", "Internal (Z-Matrix)"])
                sel_id = st.selectbox("Select Conformer ID:", [d["ID"] for d in energy_data])
                
                if coord_mode == "Cartesian (XYZ)":
                    xyz_block = Chem.MolToXYZBlock(mol_ready, confId=int(sel_id))
                    st.text_area("XYZ Coordinates", xyz_block, height=300)
                else:
                    zmat_df = get_internal_coordinates(mol_ready, int(sel_id))
                    st.dataframe(zmat_df, use_container_width=True)
                
                st.download_button("Download PDB", Chem.MolToPDBBlock(mol_ready, confId=int(sel_id)), f"conf_{sel_id}.pdb")

            with col_right:
                st.subheader("3D Visualizer")
                view = py3Dmol.view(width=600, height=400)
                view.addModel(Chem.MolToMolBlock(mol_ready, confId=int(sel_id)), 'mol')
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                view.zoomTo()
                showmol(view, height=400, width=600)

            # --- Properties Dropdown ---
            st.divider()
            category = st.selectbox("View Additional Data:", list(all_data.keys()))
            prop_cols = st.columns(len(all_data[category]))
            for i, (k, v) in enumerate(all_data[category].items()):
                prop_cols[i].metric(k, v)
