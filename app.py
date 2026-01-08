import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="Molecular Grocery List & PES Scanner", layout="wide")

st.title("Molecular Property Checker & PES Scanner")

smiles_input = st.text_input("Enter SMILES (e.g., Butane: CCCC, Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)", "CCCC").strip()

if smiles_input:
    # Validation Check
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES string. Please check the structure and try again.")
    else:
        try:
            mol = Chem.AddHs(mol)
            # Ensure 3D coordinates are generated
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:
                # Fallback for difficult molecules
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
            
            st.subheader("3D Coordinate Table")
            conf = mol.GetConformer()
            atom_data = [[a.GetSymbol(), pos.x, pos.y, pos.z] for a, pos in zip(mol.GetAtoms(), [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])]
            df_coords = pd.DataFrame(atom_data, columns=["Element", "X", "Y", "Z"])
            st.dataframe(df_coords, use_container_width=True)
            
            csv = df_coords.to_csv(index=False).encode('utf-8')
            st.download_button("Download Coordinates as CSV", data=csv, file_name='coords_table.csv', mime='text/csv')

            st.divider()

            st.subheader("Potential Energy Surface (PES) Scan")
            # Select 4 connected non-hydrogen atoms
            rotatable_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1]~[!#1]~[!#1]'))
            
            if rotatable_bonds:
                d_atoms = list(rotatable_bonds[0])
                st.info(f"Scanning Dihedral Angle for atoms: {d_atoms}")
                
                angles = np.arange(0, 370, 10)
                energies = []
                
                for angle in angles:
                    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), d_atoms[0], d_atoms[1], d_atoms[2], d_atoms[3], float(angle))
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
