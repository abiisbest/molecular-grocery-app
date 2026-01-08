import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import Draw

st.set_page_config(page_title="Molecular Grocery List & PES Scanner", layout="wide")

st.title("Molecular Property Checker & PES Scanner")

smiles_input = st.text_input("Enter SMILES (e.g., Ethanol: CCO, Butane: CCCC)", "CCCC")

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
            st.subheader("2D Structure")
            img = Draw.MolToImage(mol)
            st.image(img)

        st.divider()
        
        st.subheader("3D Coordinate Table")
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        conf = mol.GetConformer()
        
        atom_data = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_data.append([atom.GetSymbol(), pos.x, pos.y, pos.z])
        
        df_coords = pd.DataFrame(atom_data, columns=["Element", "X", "Y", "Z"])
        st.dataframe(df_coords, use_container_width=True)

        csv = df_coords.to_csv(index=False).encode('utf-8')
        st.download_button("Download Coordinates as CSV", data=csv, file_name='coords.csv', mime='text/csv')

        st.divider()

        st.subheader("Potential Energy Surface (PES) Scan")
        if mol.GetNumAtoms() >= 4:
            angles = np.arange(0, 370, 10)
            energies = []
            d_atoms = [0, 1, 2, 3] 
            
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
            
            for angle in angles:
                ff.MMFFSetDihedralDeg(d_atoms[0], d_atoms[1], d_atoms[2], d_atoms[3], float(angle))
                ff.Minimize()
                energies.append(ff.CalcEnergy())
            
            fig, ax = plt.subplots()
            ax.plot(angles, energies, marker='o', color='red')
            ax.set_xlabel("Dihedral Angle (Degrees)")
            ax.set_ylabel("Energy (kcal/mol)")
            st.pyplot(fig)
        else:
            st.error("Need at least 4 atoms for a PES scan.")

    except Exception as e:
        st.error(f"Error: {e}")
