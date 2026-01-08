import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol
import itertools

st.set_page_config(page_title="Molecular Geometry Analyzer", layout="wide")

st.title("Molecular Property Checker & Interactive 3D Analyzer")
st.write("Capstone Project: Internal Coordinates & Dynamic 3D PES")

smiles_input = st.text_input("Enter SMILES (e.g., Pentane: CCCCC, Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)", "CCCCC").strip()

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES. Please check your structure.")
    else:
        try:
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:
                AllChem.Compute2DCoords(mol)
                st.warning("Could not generate 3D coordinates; showing 2D projection.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Molecular Grocery List")
                stats = {
                    "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                    "LogP": round(Descriptors.MolLogP(mol), 2),
                    "H-Bond Donors": Descriptors.NumHDonors(mol),
                    "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                }
                st.table(pd.DataFrame(stats.items(), columns=["Property", "Value"]))

                if stats["Molecular Weight"] < 500 and stats["LogP"] < 5 and stats["H-Bond Donors"] <= 5:
                    st.success("✅ Passes Lipinski's Rule of 5")
                else:
                    st.warning("⚠️ Fails Lipinski's Rule of 5")

            with col2:
                st.subheader("3D Interactive View")
                mblock = Chem.MolToPDBBlock(mol)
                view = py3Dmol.view(width=400, height=300)
                view.addModel(mblock, 'pdb')
                view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                view.zoomTo()
                showmol(view, height=300, width=400)

            st.divider()
            
            tab1, tab2 = st.tabs(["Cartesian Coordinates (XYZ)", "Internal Coordinates (Z-Matrix)"])
            conf = mol.GetConformer()

            with tab1:
                st.subheader("3D Cartesian Coordinates")
                atom_data = [[a.GetSymbol(), i, pos.x, pos.y, pos.z] for i, (a, pos) in enumerate(zip(mol.GetAtoms(), [conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())]))]
                df_coords = pd.DataFrame(atom_data, columns=["Symbol", "Atom No.", "X", "Y", "Z"])
                st.dataframe(df_coords, use_container_width=True)

            with tab2:
                st.subheader("Internal Geometry Analysis")
                
                b_list = [[mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(), b.GetBeginAtomIdx(), 
                           mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol(), b.GetEndAtomIdx(), 
                           round(rdMolTransforms.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx()), 3)] for b in mol.GetBonds()]
                
                a_list = []
                for atom in mol.GetAtoms():
                    v_idx, v_sym = atom.GetIdx(), atom.GetSymbol()
                    nbs = [x.GetIdx() for x in atom.GetNeighbors()]
                    if len(nbs) >= 2:
                        for idx1, idx3 in itertools.combinations(nbs, 2):
                            try: ang = rdMolTransforms.GetAngleDeg(conf, idx1, v_idx, idx3)
                            except: ang = rdMolTransforms.GetBondAngleDeg(conf, idx1, v_idx, idx3)
                            a_list.append([mol.GetAtomWithIdx(idx1).GetSymbol(), idx1, v_sym, v_idx, mol.GetAtomWithIdx(idx3).GetSymbol(), idx3, round(ang, 2)])

                t_list = []
                dihedral_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1]~[!#1]~[!#1]'))
                for m in dihedral_matches:
                    t_list.append([mol.GetAtomWithIdx(m[0]).GetSymbol(), m[0], mol.GetAtomWithIdx(m[1]).GetSymbol(), m[1], 
                                   mol.GetAtomWithIdx(m[2]).GetSymbol(), m[2], mol.GetAtomWithIdx(m[3]).GetSymbol(), m[3], 
                                   round(rdMolTransforms.GetDihedralDeg(conf, m[0], m[1], m[2], m[3]), 2)])

                st.write("**1. Bond Lengths (Å)**")
                st.dataframe(pd.DataFrame(b_list, columns=["S1", "No.1", "S2", "No.2", "Length"]), use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**2. Bond Angles (°)**")
                    st.dataframe(pd.DataFrame(a_list, columns=["S1", "No.1", "Vertex", "V_No.", "S2", "No.2", "Angle"]), use_container_width=True)
                with c2:
                    st.write("**3. Twist Angles (°)**")
                    st.dataframe(pd.DataFrame(t_list, columns=["S1", "No.1", "S2", "No.2", "S3", "No.3", "S4", "No.4", "Twist"]), use_container_width=True)

            st.divider()

            st.subheader("Interactive Potential Energy Surface (PES) Scan")
            scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1&!R]~[!#1&!R]~[!#1]'))
            
            if len(scan_matches) >= 2:
                d1, d2 = list(scan_matches[0]), list(scan_matches[1])
                st.info(f"Scanning interaction: Bond {d1} vs Bond {d2}")
                
                steps = np.arange(0, 380, 20)
                Z = np.zeros((len(steps), len(steps)))
                
                pb = st.progress(0)
                total = len(steps)**2
                
                for i in range(len(steps)):
                    for j in range(len(steps)):
                        rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(steps[i]))
                        rdMolTransforms.SetDihedralDeg(conf, d2[0], d2[1], d2[2], d2[3], float(steps[j]))
                        mp = AllChem.MMFFGetMoleculeProperties(mol)
                        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                        Z[i, j] = ff.CalcEnergy() if ff else 0
                        pb.progress(((i * len(steps)) + j + 1) / total)

                # Plotly Interactive 3D Surface
                fig = go.Figure(data=[go.Surface(z=Z, x=steps, y=steps, colorscale='Viridis')])
                fig.update_layout(
                    title='Rotatable 3D Energy Surface',
                    scene = dict(
                        xaxis_title='Twist 1 (°)',
                        yaxis_title='Twist 2 (°)',
                        zaxis_title='Energy (kcal/mol)'
                    ),
                    autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif len(scan_matches) == 1:
                d1 = list(scan_matches[0])
                angles = np.arange(0, 370, 10)
                energies = []
                for a in angles:
                    rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(a))
                    mp = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                    if ff:
                        ff.Minimize(maxIts=50)
                        energies.append(ff.CalcEnergy())
                
                fig = go.Figure(data=go.Scatter(x=angles, y=energies, mode='lines+markers', line=dict(color='firebrick')))
                fig.update_layout(title='2D Energy Scan', xaxis_title='Twist Angle (°)', yaxis_title='Energy (kcal/mol)')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
