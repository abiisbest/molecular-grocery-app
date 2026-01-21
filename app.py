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
        },
        "Lead Optimization": {
            "RO5 Violations": sum([Descriptors.MolWt(mol) > 500, Descriptors.MolLogP(mol) > 5, 
                                 Lipinski.NumHDonors(mol) > 5, Lipinski.NumHAcceptors(mol) > 10]),
            "Veber Rule": "Pass" if (Lipinski.NumRotatableBonds(mol) <= 10 and Descriptors.TPSA(mol) <= 140) else "Fail"
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

st.title("Bioinformatics Analysis Platform")

st.subheader("Molecular Input")
input_mode = st.radio("Choose Input Method:", ["SMILES String", "Upload File (SDF/MOL2)"])

mol_ready_to_analyze = None
if input_mode == "SMILES String":
    smiles_text = st.text_input("Enter SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
    if smiles_text:
        mol_ready_to_analyze = Chem.MolFromSmiles(smiles_text)
else:
    up_file = st.file_uploader("Upload SDF or MOL2", type=["sdf", "mol2"])
    if up_file:
        raw_content = up_file.read().decode("utf-8")
        if up_file.name.endswith(".sdf"):
            mol_ready_to_analyze = Chem.MolFromMolBlock(raw_content)
        else:
            mol_ready_to_analyze = Chem.MolFromMol2Block(raw_content)

if mol_ready_to_analyze:
    num_conf = st.sidebar.slider("Conformers", 1, 50, 10)
    all_data = calculate_all_data(mol_ready_to_analyze)
    energy_data, mol_final = generate_conformers(mol_ready_to_analyze, num_conf)
    
    st.subheader("Potential Energy Surface (PES) Graph")
    df_pes = pd.DataFrame(energy_data).sort_values("RMSD")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pes["RMSD"], y=df_pes["Rel_E"], mode='lines+markers', line_color='teal'))
    fig.update_layout(xaxis_title="RMSD (Å)", yaxis_title="Relative Energy (kcal/mol)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Structural & Property Analysis")
    data_tab, geom_tab = st.tabs(["Molecular Data", "Geometric Coordinates"])

    with data_tab:
        category = st.selectbox("Display Data:", list(all_data.keys()))
        m_cols = st.columns(len(all_data[category]))
        for i, (k, v) in enumerate(all_data[category].items()):
            m_cols[i].metric(k, v)
        
        st.subheader("Stability Analysis")
        st.dataframe(pd.DataFrame(energy_data), use_container_width=True)

    with geom_tab:
        sel_id = st.selectbox("Select ID for Coordinates:", [d["ID"] for d in energy_data])
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.write("**Cartesian Coordinates (XYZ)**")
            st.text_area("XYZ Block", Chem.MolToXYZBlock(mol_final, confId=int(sel_id)), height=250)
        
        with g_col2:
            st.write("**Internal Coordinates (Z-Matrix)**")
            st.dataframe(get_internal_coordinates(mol_final, int(sel_id)), use_container_width=True)

    st.divider()
    st.subheader(f"3D Conformational Visualizer (ID: {sel_id})")
    view = py3Dmol.view(width=800, height=500)
    view.addModel(Chem.MolToMolBlock(mol_final, confId=int(sel_id)), 'mol')
    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
    view.zoomTo()
    showmol(view, height=500, width=800)
    st.download_button("Download PDB", Chem.MolToPDBBlock(mol_final, confId=int(sel_id)), f"conformer_{sel_id}.pdb")
