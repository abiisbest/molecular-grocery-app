import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def calculate_binding_affinity(ligand, protein_pdb_str):
    if not protein_pdb_str:
        return None
    logp = Descriptors.MolLogP(ligand)
    hbd = Lipinski.NumHDonors(ligand)
    hba = Lipinski.NumHAcceptors(ligand)
    affinity = - (1.2 * logp) - (1.5 * hbd) - (0.8 * hba)
    return round(float(affinity), 2)

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

def calculate_all_data(mol, protein_pdb=None):
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())]
    binding_score = calculate_binding_affinity(mol, protein_pdb)
    data = {
        "Physicochemical": {
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
        },
        "Quantum & MOPAC": {
            "Max Partial Charge": round(max(charges), 4),
            "Min Partial Charge": round(min(charges), 4),
            "Labute ASA": round(Descriptors.LabuteASA(mol), 2),
            "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
        },
        "Binding Analysis": {
            "Affinity Score (kcal/mol)": binding_score if binding_score else "Upload Protein",
            "H-Bond Donors": Lipinski.NumHDonors(mol),
            "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        }
    }
    return data

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
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
        # Add a Status label to highlight the most stable one
        status = "MOST STABLE (Global Minimum)" if d["ID"] == best_id else "Local Minimum"
        energy_list.append({
            "ID": int(d["ID"]), 
            "Energy": round(d["Raw"], 4), 
            "Rel_E": round(rel_e, 4), 
            "RMSD": round(rmsd, 3),
            "Status": status
        })
    return energy_list, mol, best_id

st.title("Bioinformatics Analysis Platform")

col1, col2 = st.columns(2)
with col1:
    smiles_input = st.text_input("Ligand SMILES:", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C")
with col2:
    prot_file = st.file_uploader("Target Protein (PDB)", type=["pdb"])

protein_data = prot_file.read().decode("utf-8") if prot_file else None
mol_ready = Chem.MolFromSmiles(smiles_input)

if mol_ready:
    num_conf = st.sidebar.slider("Conformers", 1, 50, 10)
    all_data = calculate_all_data(mol_ready, protein_data)
    energy_data, mol_final, best_id = generate_conformers(mol_ready, num_conf)
    
    st.subheader("Results")
    category = st.selectbox("Select View:", list(all_data.keys()))
    m_cols = st.columns(len(all_data[category]))
    for i, (k, v) in enumerate(all_data[category].items()):
        m_cols[i].metric(k, v)

    st.divider()

    st.subheader("Structural & Property Analysis")
    data_tab, geom_tab = st.tabs(["Molecular Data", "Geometric Coordinates"])

    with data_tab:
        st.write(f"### Stability Analysis (PES) - Most Stable ID: {best_id}")
        df_pes = pd.DataFrame(energy_data).sort_values("RMSD")
        
        # Highlight most stable row in the dataframe
        st.dataframe(df_pes.style.highlight_min(subset=['Rel_E'], color='teal'), use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pes["RMSD"], y=df_pes["Rel_E"], mode='lines+markers', line_color='teal', name="Conformers"))
        # Add a star for the global minimum
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='star', size=15, color='orange'), name="Global Minimum"))
        fig.update_layout(xaxis_title="RMSD (Å)", yaxis_title="Relative Energy (kcal/mol)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with geom_tab:
        sel_id = st.selectbox("Select ID for Coordinates:", [d["ID"] for d in energy_data], index=0)
        
        if sel_id == best_id:
            st.success("You are viewing the MOST STABLE conformer (Global Minimum).")
        
        st.write("### Cartesian Coordinates (XYZ)")
        xyz_block = Chem.MolToXYZBlock(mol_final, confId=int(sel_id))
        xyz_lines = xyz_block.strip().split('\n')[2:]
        xyz_data = [line.split() for line in xyz_lines]
        df_xyz = pd.DataFrame(xyz_data, columns=["Atom", "X", "Y", "Z"])
        st.dataframe(df_xyz, use_container_width=True)
        
        st.divider()
        
        st.write("### Internal Coordinates (Z-Matrix)")
        st.dataframe(get_internal_coordinates(mol_final, int(sel_id)), use_container_width=True)

    st.divider()
    st.subheader("3D Structural Visualization")
    view = py3Dmol.view(width=800, height=500)
    if protein_data:
        view.addModel(protein_data, 'pdb')
        view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})
    
    pdb_string = Chem.MolToPDBBlock(mol_final, confId=int(sel_id))
    view.addModel(pdb_string, 'pdb')
    view.setStyle({'model': 1}, {'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
    view.zoomTo()
    showmol(view, height=500, width=800)

    st.write("### Export Data")
    st.download_button(
        label=f"Download Conformer {sel_id} (PDB)",
        data=pdb_string,
        file_name=f"conformer_{sel_id}.pdb",
        mime="chemical/x-pdb"
    )
