import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures, Descriptors, Lipinski
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def get_pharmacophores(mol, conf_id):
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    return factory.GetFeaturesForMol(mol, confId=int(conf_id))

def calculate_properties(mol):
    properties = {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "RotB": Lipinski.NumRotatableBonds(mol)
    }
    violations = 0
    if properties["MW"] > 500: violations += 1
    if properties["LogP"] > 5: violations += 1
    if properties["HBD"] > 5: violations += 1
    if properties["HBA"] > 10: violations += 1
    properties["Violations"] = violations
    return properties

def generate_conformers(smiles, num_conf):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, None, None
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)
    props = calculate_properties(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5 
    
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids: return mol, props, None

    raw_data = []
    for conf_id in ids:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff:
            ff.Minimize(maxIts=1000)
            energy = ff.CalcEnergy()
            raw_data.append({"ID": conf_id, "Raw": energy})
            
    if not raw_data: return mol, props, None

    min_e = min(d["Raw"] for d in raw_data)
    final_energy_data = []
    for d in raw_data:
        relative_e = d["Raw"] - min_e
        final_energy_data.append({
            "ID": d["ID"], 
            "Rel Energy": round(relative_e, 4),
            "Stability Score": round(-relative_e, 4)
        })
    return mol, props, final_energy_data

st.title("Integrated Computational Platform for Molecular Property Prediction")
st.markdown("### Molecular Analysis & Conformational Energy Platform")

smiles_input = st.text_input("Enter 2D SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
num_conf = st.slider("Select Number of Conformers", 1, 50, 10)

if smiles_input:
    mol, mol_props, energy_data = generate_conformers(smiles_input, num_conf)
    
    if mol and mol_props:
        st.subheader("Predicted Molecular Properties")
        met_cols = st.columns(7)
        met_cols[0].metric("MW", mol_props["MW"])
        met_cols[1].metric("LogP", mol_props["LogP"])
        met_cols[2].metric("TPSA", mol_props["TPSA"])
        met_cols[3].metric("HBD", mol_props["HBD"])
        met_cols[4].metric("HBA", mol_props["HBA"])
        met_cols[5].metric("RotB", mol_props["RotB"])
        met_cols[6].metric("Violations", mol_props["Violations"])

        if energy_data:
            df = pd.DataFrame(energy_data).sort_values("Rel Energy")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Stability Analysis")
                st.dataframe(df[["ID", "Stability Score"]], use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "molecular_data.csv", "text/csv")
                
                st.divider()
                selected_id = st.selectbox("Select Conformer ID", df["ID"])
                st.download_button("Download PDB", Chem.MolToPDBBlock(mol, confId=int(selected_id)), f"conf_{selected_id}.pdb", "chemical/x-pdb")
                
                st.markdown("#### Legend")
                st.write("🔵 **Donor** | 🔴 **Acceptor** | 🟠 **Aromatic**")
                
            with col2:
                st.subheader(f"3D Visualizer (ID: {selected_id})")
                feats = get_pharmacophores(mol, selected_id)
                view = py3Dmol.view(width=800, height=500)
                view.addModel(Chem.MolToMolBlock(mol, confId=int(selected_id)), 'mol')
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                
                for f in feats:
                    p = f.GetPos(int(selected_id)) 
                    c = "blue" if f.GetFamily()=="Donor" else "red" if f.GetFamily()=="Acceptor" else "orange"
                    view.addSphere({'center':{'x':p.x,'y':p.y,'z':p.z}, 'radius':0.7, 'color':c, 'opacity':0.5})
                
                view.zoomTo()
                showmol(view, height=500, width=800)
    else:
        st.error("Invalid SMILES.")
