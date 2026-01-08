import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol
import itertools

st.set_page_config(page_title="Molecular Geometry Analyzer", layout="wide")

st.title("Molecular Property Checker & Geometry Analyzer")
st.write("A Capstone Project for Molecular Modelling")

smiles_input = st.text_input("Enter SMILES (e.g., Butane: CCCC, Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)", "CCCC").strip()

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES string. Please check the structure and try again.")
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
                    st.warning("⚠️ Fails one or more Lipinski Rules")

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
                df_coords = pd.DataFrame(atom_data, columns=["Chemical Symbol", "Atom Number", "X", "Y", "Z"])
                st.dataframe(df_coords, use_container_width=True)
                
                csv_xyz = df_coords.to_csv(index=False).encode('utf-8')
                st.download_button("Download XYZ as CSV", data=csv_xyz, file_name='cartesian_coords.csv', mime='text/csv')

            with tab2:
                st.subheader("Internal Geometry Analysis")
                
                # 1. Bond Lengths
                bonds_list = []
                for bond in mol.GetBonds():
                    i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    sym1, sym2 = mol.GetAtomWithIdx(i1).GetSymbol(), mol.GetAtomWithIdx(i2).GetSymbol()
                    length = rdMolTransforms.GetBondLength(conf, i1, i2)
                    bonds_list.append([sym1, i1, sym2, i2, round(length, 3)])
                
                # 2. Bond Angles
                angles_list = []
                for atom in mol.GetAtoms():
                    v_idx = atom.GetIdx()
                    v_sym = atom.GetSymbol()
                    nbs = [x.GetIdx() for x in atom.GetNeighbors()]
                    if len(nbs) >= 2:
                        for idx1, idx3 in itertools.combinations(nbs, 2):
                            s1, s3 = mol.GetAtomWithIdx(idx1).GetSymbol(), mol.GetAtomWithIdx(idx3).GetSymbol()
                            try: ang = rdMolTransforms.GetAngleDeg(conf, idx1, v_idx, idx3)
                            except: ang = rdMolTransforms.GetBondAngleDeg(conf, idx1, v_idx, idx3)
                            angles_list.append([s1, idx1, v_sym, v_idx, s3, idx3, round(ang, 2)])

                # 3. Twist Angles (Dihedrals)
                twist_list = []
                dihedral_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]~[*]~[*]~[*]'))
                for m in dihedral_matches:
                    syms = [mol.GetAtomWithIdx(x).GetSymbol() for x in m]
                    twist = rdMolTransforms.GetDihedralDeg(conf, m[0], m[1], m[2], m[3])
                    twist_list.append([syms[0], m[0], syms[1], m[1], syms[2], m[2], syms[3], m[3], round(twist, 2)])

                st.write("**1. Bond Lengths**")
                st.dataframe(pd.DataFrame(bonds_list, columns=["Symbol 1", "Atom No. 1", "Symbol 2", "Atom No. 2", "Bond Length (Å)"]), use_container_width=True)
                
                col_a, col_t = st.columns(2)
                with col_a:
                    st.write("**2. Bond Angles**")
                    st.dataframe(pd.DataFrame(angles_list, columns=["S1", "No.1", "Vertex Sym", "Vertex No.", "S2", "No.2", "Bond Angle (°)"]), use_container_width=True)
                with col_t:
                    st.write("**3. Twist Angles (Dihedrals)**")
                    st.dataframe(pd.DataFrame(twist_list, columns=["S1", "No.1", "S2", "No.2", "S3", "No.3", "S4", "No.4", "Twist Angle (°)"]), use_container_width=True)

            st.divider()

            st.subheader("Potential Energy Surface (PES) Scan")
            # Selection priority: Non-ring bonds first, then any 4-atom chain
            scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1&!R]~[!#1&!R]~[!#1]'))
            if not scan_matches:
                scan_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]~[*]~[*]~[*]'))

            if scan_matches:
                d_atoms = list(scan_matches[0])
                st.info(f"Scanning Twist Angle for atoms: {d_atoms}")
                
                angles = np.arange(0, 370, 10)
                energies = []
                prog_bar = st.progress(0)
                
                for i, angle in enumerate(angles):
                    rdMolTransforms.SetDihedralDeg(conf, d_atoms[0], d_atoms[1], d_atoms[2], d_atoms[3], float(angle))
                    mp = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                    if ff:
                        # Only minimize if center bond is not in a ring
                        bond = mol.GetBondBetweenAtoms(d_atoms[1], d_atoms[2])
                        if bond and not bond.IsInRing():
                            ff.Minimize(maxIts=50)
                        energies.append(ff.CalcEnergy())
                    else:
                        energies.append(np.nan)
                    prog_bar.progress((i + 1) / len(angles))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(angles, energies, marker='o', color='#FF4B4B', linewidth=2)
                ax.set_xlabel("Twist Angle (Degrees)")
                ax.set_ylabel("Energy (kcal/mol)")
                ax.set_title("Energy vs. Dihedral Rotation")
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
            else:
                st.warning("Molecule size insufficient for PES scan.")

            st.divider()
            with st.expander("Glossary & Theory"):
                st.write("""
                - **Internal Coordinates**: Describes molecular shape using Bond Lengths, Bond Angles, and Twist (Dihedral) Angles.
                - **Z-Matrix**: A classical representation of molecules where each atom is defined by its relation to previous atoms.
                - **Å (Angstrom)**: $10^{-10}$ meters, the standard unit for atomic distances.
                - **PES Scan**: A Potential Energy Surface scan helps find the most stable (lowest energy) conformation of a molecule.
                """)

        except Exception as e:
            st.error(f"Error processing molecule: {e}")
