import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol
import itertools

st.set_page_config(page_title="Molecular Intelligence Platform", layout="wide")

st.title("Integrated Computational Platform for Molecular Property & Conformational Analysis")
st.write("Capstone Project: Advanced Molecular Modeling & Dynamic Energy Landscapes")

smiles_input = st.text_input("Enter SMILES (e.g., Pentane: CCCCC, Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)", "CC(=O)OC1=CC=CC=C1C(=O)O").strip()

def get_bioisosteres(mol):
    replacements = {
        "C(=O)O": ["c1nn[nH]n1", "S(=O)(=O)O"],
        "C(=O)N": ["C(=S)N", "c1nno[nH]1"],
        "OH": ["F", "NH2", "SH"],
        "Cl": ["CF3", "CH3"],
    }
    suggestions = []
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    for pattern, reps in replacements.items():
        if pattern in smiles:
            for r in reps:
                suggestions.append(smiles.replace(pattern, r))
    return list(set(suggestions))

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
            
            conf = mol.GetConformer()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Molecular Property Ensemble")
                stats = {
                    "Molecular Weight": round(Descriptors.MolWt(mol), 2),
                    "LogP": round(Descriptors.MolLogP(mol), 2),
                    "H-Bond Donors": Descriptors.NumHDonors(mol),
                    "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                    "TPSA": round(Descriptors.TPSA(mol), 2)
                }
                st.table(pd.DataFrame(stats.items(), columns=["Property", "Value"]))

                if stats["Molecular Weight"] <= 500 and stats["LogP"] <= 5 and stats["H-Bond Donors"] <= 5:
                    st.success("✅ Passes Lipinski's Rule of 5")
                else:
                    st.warning("⚠️ Fails Lipinski's Rule of 5")
                
                st.subheader("Lead Optimization (Bioisosteres)")
                fixes = get_bioisosteres(mol)
                if fixes:
                    st.info("Bioisosteric swaps for structural optimization:")
                    for f in fixes: st.code(f)
                else:
                    st.write("No standard swaps identified.")

            with col2:
                st.subheader("3D Electronic Hotspot Map")
                AllChem.ComputeGasteigerCharges(mol)
                mblock = Chem.MolToPDBBlock(mol)
                view = py3Dmol.view(width=400, height=400)
                view.addModel(mblock, 'pdb')
                
                for i, atom in enumerate(mol.GetAtoms()):
                    chg = float(atom.GetProp('_GasteigerCharge'))
                    color = "red" if chg < -0.06 else "blue" if chg > 0.06 else "white"
                    view.setStyle({'serial': i}, {'stick': {'color': color}, 'sphere': {'scale': 0.3, 'color': color}})
                
                view.zoomTo()
                showmol(view, height=400, width=400)
                st.caption("Red: Nucleophilic (-) | Blue: Electrophilic (+) | White: Neutral")

            st.divider()
            
            tab1, tab2 = st.tabs(["Cartesian Coordinates (XYZ)", "Internal Geometry (Z-Matrix)"])

            with tab1:
                atom_data = [[a.GetSymbol(), i, pos.x, pos.y, pos.z] for i, (a, pos) in enumerate(zip(mol.GetAtoms(), [conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())]))]
                st.dataframe(pd.DataFrame(atom_data, columns=["Symbol", "Atom No.", "X", "Y", "Z"]), use_container_width=True)

            with tab2:
                b_list = [[mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(), b.GetBeginAtomIdx(), mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol(), b.GetEndAtomIdx(), round(rdMolTransforms.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx()), 3)] for b in mol.GetBonds()]
                a_list = []
                for atom in mol.GetAtoms():
                    v_idx, v_sym = atom.GetIdx(), atom.GetSymbol()
                    nbs = [x.GetIdx() for x in atom.GetNeighbors()]
                    if len(nbs) >= 2:
                        for idx1, idx3 in itertools.combinations(nbs, 2):
                            ang = rdMolTransforms.GetAngleDeg(conf, idx1, v_idx, idx3)
                            a_list.append([mol.GetAtomWithIdx(idx1).GetSymbol(), idx1, v_sym, v_idx, mol.GetAtomWithIdx(idx3).GetSymbol(), idx3, round(ang, 2)])
                
                st.write("**Bond Lengths & Angles**")
                st.dataframe(pd.DataFrame(b_list, columns=["S1", "No.1", "S2", "No.2", "Length"]), use_container_width=True)
                st.dataframe(pd.DataFrame(a_list, columns=["S1", "No.1", "Vertex", "V_No.", "S2", "No.2", "Angle"]), use_container_width=True)

            st.divider()

            st.subheader("Potential Energy Surface (PES) Scan")
            scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1&!R]~[!#1&!R]~[!#1]'))
            
            if len(scan_matches) >= 2:
                d1, d2 = list(scan_matches[0]), list(scan_matches[1])
                st.info(f"Scanning Dihedrals: {d1} and {d2}")
                steps = np.arange(0, 380, 30)
                Z = np.zeros((len(steps), len(steps)))
                
                for i in range(len(steps)):
                    for j in range(len(steps)):
                        rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(steps[i]))
                        rdMolTransforms.SetDihedralDeg(conf, d2[0], d2[1], d2[2], d2[3], float(steps[j]))
                        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
                        Z[i, j] = ff.CalcEnergy() if ff else 0

                fig = go.Figure(data=[go.Surface(z=Z, x=steps, y=steps, colorscale='Viridis')])
                fig.update_layout(title='3D PES Map', scene=dict(xaxis_title='Torsion 1', yaxis_title='Torsion 2', zaxis_title='Energy'), width=800, height=700)
                st.plotly_chart(fig, use_container_width=True)
                
            elif len(scan_matches) == 1:
                d1 = list(scan_matches[0])
                angles = np.arange(0, 370, 10)
                energies = []
                for a in angles:
                    rdMolTransforms.SetDihedralDeg(conf, d1[0], d1[1], d1[2], d1[3], float(a))
                    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
                    energies.append(ff.CalcEnergy() if ff else 0)
                
                fig = go.Figure(data=go.Scatter(x=angles, y=energies, mode='lines+markers'))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Computation Error: {e}")
