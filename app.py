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
    
    # Estimation of Binding Affinity (Delta G) 
    # Based on Hydrophobicity (LogP) and Hydrogen Bonding potential
    logp = Descriptors.MolLogP(ligand)
    hbd = Lipinski.NumHDonors(ligand)
    hba = Lipinski.NumHAcceptors(ligand)
    mw = Descriptors.MolWt(ligand)
    
    # Simple empirical scoring formula for affinity estimation
    affinity = - (0.5 * logp) - (0.1 * hbd) - (0.05 * hba) - (0.001 * mw)
    return round(affinity, 2)

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
        "Binding Affinity": {
            "Score (kcal/mol)": binding_score if binding_score else "Upload Protein",
            "H-Bond Donors": Lipinski.NumHDonors(mol),
            "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        }
    }
    return data

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
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

col_a, col_b = st.columns(2)
with col_a:
    smiles_input = st.text_input("Ligand SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
with col_b:
    prot_file = st.file_uploader("Target Protein (PDB)", type=["pdb"])

protein_data = prot_file.read().decode("utf-8") if prot_file else None
mol_ready = Chem.MolFromSmiles(smiles_input)

if mol_ready:
    num_conf = st.sidebar.slider("Conformers", 1, 50, 10)
    all_data = calculate_all_data(mol_ready, protein_data)
    energy_data, mol_final = generate_conformers(mol_ready, num_conf)
    
    st.subheader("Results")
    category = st.selectbox("Select View:", list(all_data.keys()))
    m_cols = st.columns(len(all_data[category]))
    for i, (k, v) in enumerate(all_data[category].items()):
        m_cols[i].metric(k, v)

    st.subheader("Complex Visualization")
    sel_id = st.selectbox("Ligand Conformer ID:", [d["ID"] for d in energy_data])
    view = py3Dmol.view(width=800, height=500)
    if protein_data:
        view.addModel(protein_data, 'pdb')
        view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})
    view.addModel(Chem.MolToMolBlock(mol_final, confId=int(sel_id)), 'mol')
    view.setStyle({'model': 1}, {'stick': {}})
    view.zoomTo()
    showmol(view, height=500, width=800)
