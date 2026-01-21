import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, ChemicalFeatures, Descriptors, Lipinski, FilterCatalog
import py3Dmol
from stmol import showmol
import pandas as pd
import os

st.set_page_config(page_title="Bioinformatics Analysis Platform", layout="wide")

def check_pains(mol):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return "Detected" if catalog.HasMatch(mol) else "Clean"

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
        "RotB": Lipinski.NumRotatableBonds(mol),
        "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
        "PAINS": check_pains(mol)
    }
    violations = 0
    if properties["MW"] > 500: violations += 1
    if properties["LogP"] > 5: violations += 1
    if properties["HBD"] > 5: violations += 1
    if properties["HBA"] > 10: violations += 1
    properties["RO5 Violations"] = violations
    return properties

def generate_conformers(mol, num_conf):
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5 
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    if not ids: return None, None
    
    raw_data = []
    for conf_id in ids:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff:
            ff.Minimize(maxIts=1000)
            energy = ff.CalcEnergy() 
            raw_data.append({"ID": conf_id, "Raw": energy})
    
    if not raw_data: return None, None
    
    min_e = min(d["Raw"] for d in raw_data)
    best_id = min(raw_data, key=lambda x: x["Raw"])["ID"]
    
    energy_list = []
    for d in raw_data:
        rel_e = d["Raw"] - min_e
        rmsd = AllChem.GetConformerRMS(mol, best_id, d["ID"])
        energy_list.append({
            "ID": int(d["ID"]), 
            "Energy (kcal/mol)": round(d["Raw"], 4),
            "Stability Score (Rel)": round(-rel_e, 4), 
            "RMSD (Å)": round(rmsd, 3)
        })
    return energy_list, mol

st.title("Integrated Computational Platform for Molecular Property Prediction")

tab1, tab2 = st.tabs(["Single Molecule Analysis", "Batch Screening"])

with tab1:
    smiles_input = st.text_input("Enter 2D SMILES:", "CC(C)c1c(c(c(n1CC[C@H](C[C@H](CC(=O)O)O)O)c2ccc(cc2)F)c3ccccc3)C(=O)Nc4ccccc4")
    num_conf = st.slider("Conformers to Generate", 1, 50, 10, key="single_slider")
    
    if smiles_input:
        mol_base = Chem.MolFromSmiles(smiles_input)
        if mol_base:
            props = calculate_properties(mol_base)
            energy_data, mol_ready = generate_conformers(mol_base, num_conf)
            
            st.subheader("Molecular Properties")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MW", props["MW"])
            c2.metric("LogP", props["LogP"])
            c3.metric("TPSA", props["TPSA"])
            c4.metric("PAINS", props["PAINS"])
            c5.metric("RO5 Violations", props["RO5 Violations"])
            
            if energy_data:
                df = pd.DataFrame(energy_data).sort_values("Stability Score (Rel)", ascending=False)
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.subheader("Conformer Stability")
                    st.dataframe(df[["ID", "Stability Score (Rel)", "RMSD (Å)"]], use_container_width=True)
                    
                    sel_id = st.selectbox("Select ID for 3D View", df["ID"].tolist())
                    
                    if sel_id is not None:
                        pdb_data = Chem.MolToPDBBlock(mol_ready, confId=int(sel_id))
                        st.download_button("Download PDB", pdb_data, f"conf_{sel_id}.pdb")

                with col_right:
                    st.subheader(f"3D Visualizer (ID: {sel_id})")
                    feats = get_pharmacophores(mol_ready, sel_id)
                    view = py3Dmol.view(width=800, height=500)
                    view.addModel(Chem.MolToMolBlock(mol_ready, confId=int(sel_id)), 'mol')
                    view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
                    for f in feats:
                        p = f.GetPos(int(sel_id))
                        col = "blue" if f.GetFamily()=="Donor" else "red" if f.GetFamily()=="Acceptor" else "orange"
                        view.addSphere({'center':{'x':p.x,'y':p.y,'z':p.z}, 'radius':0.7, 'color':col, 'opacity':0.5})
                    view.zoomTo()
                    showmol(view, height=500, width=800)
                    st.write("🔵 **Donor** | 🔴 **Acceptor** | 🟠 **Aromatic**")

with tab2:
    st.subheader("High-Throughput Batch Screening")
    uploaded_file = st.file_uploader("Upload CSV with 'SMILES' column", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if "SMILES" in batch_df.columns:
            results = []
            for sm in batch_df["SMILES"]:
                m = Chem.MolFromSmiles(str(sm))
                if m:
                    p = calculate_properties(m)
                    p["SMILES"] = sm
                    results.append(p)
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            st.download_button("Download Batch Results (CSV)", res_df.to_csv(index=False).encode('utf-8'), "batch_results.csv")
