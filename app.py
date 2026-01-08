import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="Molecular Grocery List & PES Scanner", layout="wide")

st.title("Molecular Property Checker & PES Scanner")

smiles_input = st.text_input("Enter SMILES (e.g., Butane: CCCC, Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)", "CCCC")

if smiles_input:
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        mol = Chem.AddHs(mol)
        
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

        with col2:
            st.subheader("3D Interactive View")
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
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
        st.download_button("Download Coordinates as CSV", data=csv, file_name='coords.csv', mime='text/csv')

        st.divider()

        st.subheader("Potential Energy Surface (PES) Scan")
        rotatable_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[!#1]~[!#1]~[!#1]~[!#1]'))
        
        if rotatable_bonds:
            d_atoms = list(rotatable_bonds[0])
            st.info(f"Scanning Dihedral Angle for atoms: {d_atoms}")
            
            angles = np.arange(0, 370, 10)
            energies = []
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            
            for angle in angles:
                ff.MMFFSetDihedralDeg(d_atoms[0], d_atoms[1], d_atoms[2], d_atoms[3], float(angle))
                ff.Minimize()
                energies.append(ff.CalcEnergy())
            
            fig, ax = plt.subplots()
            ax.plot(angles, energies, marker='o', color='#FF4B4B')
            ax.set_xlabel("Dihedral Angle (Degrees)")
            ax.set_ylabel("Energy (kcal/mol)")
            st.pyplot(fig)
        else:
            st.warning("No rotatable dihedral bonds (4 connected non-hydrogen atoms) found for PES scan.")

    except Exception as e:
        st.error(f"Error: {e}")
