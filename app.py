import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol
import itertools

st.set_page_config(page_title="Molecular Grocery List & PES Scanner", layout="wide")

st.title("Molecular Property Checker & PES Scanner")

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

                pass_rules = stats["Molecular Weight"] < 500 and stats["LogP"] < 5 and stats["H-Bond Donors"] <= 5
                if pass_rules:
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
            
            tab1, tab2 = st.tabs(["Cartesian Coordinates (XYZ)", "Internal Coordinates"])
            
            conf = mol.GetConformer()

            with tab1:
                st.subheader("3D Cartesian Coordinate Table")
                atom_data = [[a.GetSymbol(), i, pos.x, pos.y, pos.z] for i, (a, pos) in enumerate(zip(mol.GetAtoms(), [conf.GetAtomPosition(k) for k in range(mol.GetNumAtoms())]))]
                df_coords = pd.DataFrame(atom_data, columns=["Element", "Index", "X", "Y", "Z"])
                st.dataframe(df_coords, use_container_width=True)
                
                csv_xyz = df_coords.to_csv(index=False).encode('utf-8')
                st.download_button("Download XYZ as CSV", data=csv_xyz, file_name='cartesian_coords.csv', mime='text/csv')

            with tab2:
                st.subheader("Internal Coordinates")
                
                # Bond Lengths
                bonds_list = []
                for bond in mol.GetBonds():
                    idx1 = bond.GetBeginAtomIdx()
                    idx2 = bond.GetEndAtomIdx()
                    length = rdMolTransforms.GetBondLength(conf, idx1, idx2)
                    bonds_list.append([
                        f"{mol.GetAtomWithIdx(idx1).GetSymbol()}({idx1})",
                        f"{mol.GetAtomWithIdx(idx2).GetSymbol()}({idx2})",
                        round(length, 3)
                    ])
                
                # Bond Angles (Calculated manually to avoid RDKit version errors)
                angles_list = []
                for atom in mol.GetAtoms():
                    idx2 = atom.GetIdx()
                    neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                    if len(neighbors) >= 2:
                        for idx1, idx3 in itertools.combinations(neighbors, 2):
                            # Using GetAngleDeg which is the more common attribute in RDKit
                            try:
                                angle = rdMolTransforms.GetAngleDeg(conf, idx1, idx2, idx3)
                            except AttributeError:
                                # Fallback if even GetAngleDeg fails
                                angle = rdMolTransforms.GetBondAngleDeg(conf, idx1, idx2, idx3)
                            
                            angles_list.append([
                                f"{mol.GetAtomWithIdx(idx1).GetSymbol()}({idx1})",
                                f"{mol.GetAtomWithIdx(idx2).GetSymbol()}({idx2})",
                                f"{mol.GetAtomWithIdx(idx3).GetSymbol()}({idx3})",
                                round(angle, 2)
                            ])

                col_b, col_a = st.columns(2)
                with col_b:
                    st.write("**Bond Lengths (Å)**")
                    st.dataframe(pd.DataFrame(bonds_list, columns=["Atom 1", "Atom 2", "Length"]), use_container_width=True)
                with col_a:
                    st.write("**Bond Angles (°)**")
                    st.dataframe(pd.DataFrame(angles_list, columns=["Atom 1", "Vertex", "Atom 2", "Angle"]), use_container_width=True)

            st.divider()

            st.subheader("Potential Energy Surface (PES) Scan")
            rotatable_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1]~[!#1]~[!#1]'))
            
            if rotatable_bonds:
                d_atoms = list(rotatable_bonds[0])
                st.info(f"Scanning Dihedral Angle for atoms: {d_atoms}")
                
                angles = np.arange(0, 370, 10)
                energies = []
                
                for angle in angles:
                    rdMolTransforms.SetDihedralDeg(conf, d_atoms[0], d_atoms[1], d_atoms[2], d_atoms[3], float(angle))
                    mp = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                    if ff:
                        energies.append(ff.CalcEnergy())
                    else:
                        energies.append(np.nan)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(angles, energies, marker='o', linestyle='-', color='#FF4B4B')
                ax.set_xlabel("Dihedral Angle (Degrees)")
                ax.set_ylabel("Energy (kcal/mol)")
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
            else:
                st.warning("No rotatable dihedral bonds found for PES scan.")

        except Exception as e:
            st.error(f"Error: {e}")
