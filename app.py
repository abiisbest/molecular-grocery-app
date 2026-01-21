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

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; }
    .stTabs [aria-selected="true"] { background-color: #4e79a7; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.info("This integrated platform provides high-fidelity molecular analysis for drug discovery and structural biology.")
    st.subheader("Quick Guide")
    st.write("1. Enter a **SMILES** string.")
    st.write("2. Navigate through the **Tabs** for specific analyses.")
    st.write("3. Export data from the **Geometry** tab.")

# --- APP HEADER ---
st.title("Integrated Computational Platform for Molecular Intelligence")
smiles_input = st.text_input("Enter SMILES String (e.g., Aspirin, Caffeine, or a custom chain):", "CC(=O)OC1=CC=CC=C1C(=O)O").strip()

def get_bioisosteres(mol):
    replacements = {
        "C(=O)O": ["c1nn[nH]n1 (Tetrazole)", "S(=O)(=O)O (Sulfonic Acid)"],
        "C(=O)N": ["C(=S)N (Thioamide)", "c1nno[nH]1 (Oxadiazole)"],
        "OH": ["F (Fluorine)", "NH2 (Amine)", "SH (Thiol)"],
        "Cl": ["CF3 (Trifluoromethyl)", "CH3 (Methyl)"],
    }
    suggestions = []
    clean_mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(clean_mol)
    for pattern, reps in replacements.items():
        if pattern in smiles:
            for r in reps: suggestions.append(smiles.replace(pattern, r))
    return list(set(suggestions))

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES structure. Please verify the input.")
    else:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            conf = mol.GetConformer()

            # --- TABBED INTERFACE ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Property Analytics", 
                "⚛️ 3D Hotspot Mapping", 
                "📐 Geometry & Export", 
                "🔋 Energy Landscapes"
            ])

            # TAB 1: PROPERTY ANALYTICS
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Physicochemical Profile")
                    stats = {
                        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                        "LogP (Hydrophobicity)": round(Descriptors.MolLogP(mol), 2),
                        "H-Bond Donors": Descriptors.NumHDonors(mol),
                        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
                        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
                    }
                    st.table(pd.DataFrame(stats.items(), columns=["Property", "Value"]))
                    
                    if stats["Molecular Weight"] <= 500 and stats["LogP (Hydrophobicity)"] <= 5:
                        st.success("✅ Favorable Drug-Likeness (Lipinski)")
                    else:
                        st.warning("⚠️ Non-ideal pharmacokinetics predicted.")

                with col2:
                    st.subheader("Structural Optimization Suggestions")
                    fixes = get_bioisosteres(mol)
                    if fixes:
                        st.write("Consider these Bioisosteric Swaps to improve stability/potency:")
                        for f in fixes: st.code(f)
                    else:
                        st.write("No common bioisosteric replacements found for this scaffold.")

            # TAB 2: 3D HOTSPOT MAPPING
            with tab2:
                st.subheader("Electronic Potential Mapping")
                st.write("Visualizing reactive sites via Gasteiger Partial Charges.")
                AllChem.ComputeGasteigerCharges(mol)
                mblock = Chem.MolToPDBBlock(mol)
                view = py3Dmol.view(width=800, height=500)
                view.addModel(mblock, 'pdb')
                
                for i, atom in enumerate(mol.GetAtoms()):
                    chg = float(atom.GetProp('_GasteigerCharge'))
                    color = "red" if chg < -0.06 else "blue" if chg > 0.06 else "white"
                    view.setStyle({'serial': i}, {'stick': {'color': color}, 'sphere': {'scale': 0.3, 'color': color}})
                
                view.zoomTo()
                showmol(view, height=500, width=800)
                st.caption("🔴 Negative Charge (Nucleophilic) | 🔵 Positive Charge (Electrophilic) | ⚪ Neutral")

            # TAB 3: GEOMETRY & EXPORT
            with tab3:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Cartesian Coordinates (XYZ)")
                    atom_data = [[a.GetSymbol(), i, pos.x, pos.y, pos.z] for i, (a, pos) in enumerate(zip(mol.GetAtoms(), [conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())]))]
                    df_xyz = pd.DataFrame(atom_data, columns=["Atom", "ID", "X", "Y", "Z"])
                    st.dataframe(df_xyz, use_container_width=True)
                    csv = df_xyz.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download XYZ CSV", csv, "geometry.csv", "text/csv")
                
                with col_b:
                    st.subheader("Internal Geometry (Bond Details)")
                    b_list = [[mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(), b.GetBeginAtomIdx(), mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol(), b.GetEndAtomIdx(), round(rdMolTransforms.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx()), 3)] for b in mol.GetBonds()]
                    st.dataframe(pd.DataFrame(b_list, columns=["S1", "ID1", "S2", "ID2", "Length (Å)"]), use_container_width=True)

            # TAB 4: ENERGY LANDSCAPES
            with tab4:
                st.subheader("Potential Energy Surface (PES) Scan")
                scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1&!R]~[!#1&!R]~[!#1]'))
                
                if len(scan_matches) >= 2:
                    st.info("Double Dihedral Scan: Visualizing molecular flexibility.")
                    d1, d2 = list(scan_matches[0]), list(scan_matches[1])
                    steps = np.arange(0, 380, 40)
                    Z = np.zeros((len(steps), len(steps)))
                    
                    for i, ang1 in enumerate(steps):
                        for j, ang2 in enumerate(steps):
                            rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(ang1))
                            rdMolTransforms.SetDihedralDeg(conf, d2[0], d2[1], d2[2], d2[3], float(ang2))
                            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
                            Z[i, j] = ff.CalcEnergy() if ff else 0

                    fig = go.Figure(data=[go.Surface(z=Z, x=steps, y=steps, colorscale='Viridis')])
                    fig.update_layout(scene=dict(xaxis_title='Twist 1', yaxis_title='Twist 2', zaxis_title='Energy (kcal/mol)'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("The molecule requires more rotatable bonds for a 3D Energy Surface Scan.")

        except Exception as e:
            st.error(f"Computation Error: {e}")
