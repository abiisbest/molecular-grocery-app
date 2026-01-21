import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol
import itertools

st.set_page_config(page_title="Bio-Chem Research Suite", layout="wide")

# --- SIDEBAR: EDUCATIONAL SUITE ---
with st.sidebar:
    st.header("🔬 Researcher's Handbook")
    st.info("""
    **LogP:** Measures how well a drug dissolves in fats vs. water. High LogP = better absorption but harder to excrete.
    
    **TPSA:** Polar Surface Area. Values < 140 Å² are usually needed for cell penetration.
    
    **PES Scan:** Shows how 'flexible' a molecule is. Deep valleys represent the most stable shapes (conformers).
    """)
    st.divider()
    st.subheader("Reference Standards")
    st.write("- **Lipinski's Rule:** MW < 500, LogP < 5, HBD < 5, HBA < 10.")

# --- MAIN INTERFACE ---
st.title("Integrated Molecular Modeling & Property Prediction Platform")
st.write("A comprehensive computational suite for lead optimization and conformational analysis.")

smiles_input = st.text_input("Input SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")

def get_bioisosteres(mol):
    replacements = {
        "C(=O)O": ["c1nn[nH]n1", "S(=O)(=O)O"],
        "C(=O)N": ["C(=S)N", "c1nno[nH]1"],
        "OH": ["F", "NH2", "SH"],
        "Cl": ["CF3", "CH3"],
    }
    suggestions = []
    clean_mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(clean_mol)
    for pattern, reps in replacements.items():
        if pattern in smiles:
            for r in reps:
                suggestions.append(smiles.replace(pattern, r))
    return list(set(suggestions))

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES structure.")
    else:
        try:
            # 1. PREPARATION
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            conf = mol.GetConformer()
            
            # 2. ANALYSIS COLUMNS
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("📋 Molecular Descriptors")
                stats = {
                    "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                    "LogP (Hydrophobicity)": round(Descriptors.MolLogP(mol), 2),
                    "H-Bond Donors": Descriptors.NumHDonors(mol),
                    "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                    "TPSA (Å²)": round(Descriptors.TPSA(mol), 2)
                }
                df_stats = pd.DataFrame(stats.items(), columns=["Property", "Value"])
                st.table(df_stats)

                # Drug-likeness Validation
                is_drug_like = (stats["Molecular Weight"] <= 500 and stats["LogP (Hydrophobicity)"] <= 5)
                if is_drug_like:
                    st.success("✅ Favorable Drug-Likeness Profile")
                else:
                    st.warning("⚠️ Molecule may face bioavailability issues.")

                st.subheader("💡 Structural Optimization")
                fixes = get_bioisosteres(mol)
                if fixes:
                    st.write("Suggested Bioisosteres to optimize potency/solubility:")
                    for f in fixes: st.code(f)
                else:
                    st.write("No common lead-optimization swaps found.")

            with col2:
                st.subheader("⚛️ 3D Electronic Mapping")
                AllChem.ComputeGasteigerCharges(mol)
                mblock = Chem.MolToPDBBlock(mol)
                view = py3Dmol.view(width=500, height=400)
                view.addModel(mblock, 'pdb')
                
                for i, atom in enumerate(mol.GetAtoms()):
                    chg = float(atom.GetProp('_GasteigerCharge'))
                    color = "red" if chg < -0.06 else "blue" if chg > 0.06 else "white"
                    view.setStyle({'serial': i}, {'stick': {'color': color}, 'sphere': {'scale': 0.3, 'color': color}})
                
                view.zoomTo()
                showmol(view, height=400, width=500)
                st.caption("Visualizing Electrophilic (Blue) and Nucleophilic (Red) centers.")

            # 3. COORDINATE DATA & EXPORT
            st.divider()
            tab1, tab2 = st.tabs(["Coordinate Systems", "Energy Landscape (PES)"])

            with tab1:
                st.subheader("Geometry Data")
                atom_data = [[a.GetSymbol(), i, pos.x, pos.y, pos.z] for i, (a, pos) in enumerate(zip(mol.GetAtoms(), [conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())]))]
                df_coords = pd.DataFrame(atom_data, columns=["Symbol", "Atom_No", "X", "Y", "Z"])
                
                st.dataframe(df_coords, use_container_width=True)
                
                # Export Button for Researchers
                csv = df_coords.to_csv(index=False).encode('utf-8')
                st.download_button("Download XYZ Coordinates as CSV", csv, "molecule_coords.csv", "text/csv")

            with tab2:
                st.subheader("Potential Energy Surface")
                scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1&!R]~[!#1&!R]~[!#1]'))
                
                if len(scan_matches) >= 2:
                    d1, d2 = list(scan_matches[0]), list(scan_matches[1])
                    steps = np.arange(0, 380, 40)
                    Z = np.zeros((len(steps), len(steps)))
                    
                    for i, ang1 in enumerate(steps):
                        for j, ang2 in enumerate(steps):
                            rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(ang1))
                            rdMolTransforms.SetDihedralDeg(conf, d2[0], d2[1], d2[2], d2[3], float(ang2))
                            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
                            Z[i, j] = ff.CalcEnergy() if ff else 0

                    fig = go.Figure(data=[go.Surface(z=Z, x=steps, y=steps, colorscale='Plasma')])
                    fig.update_layout(scene=dict(xaxis_title='Dihedral 1', yaxis_title='Dihedral 2', zaxis_title='Energy (kcal/mol)'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("The molecule is too rigid or small for a 3D PES Scan. Add a chain of at least 4 non-ring atoms.")

        except Exception as e:
            st.error(f"Platform Error: {e}. Try a different SMILES string.")
