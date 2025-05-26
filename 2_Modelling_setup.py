
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rdkit import Chem, DataStructs
from rdkit.Chem import Mol, PandasTools, AllChem, Draw, rdMolDescriptors
# from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import sklearn
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import cirpy
import pandas as pd

def generate_unhashed_fingerprints(data, radius=2):
    fps = []

    # countSimulation=True → get a count-based (unhashed) fingerprint
    fp_generator = GetMorganGenerator(
        radius=radius,
        countSimulation=True,           # ← this is key!
        onlyNonzeroInvariants=True
    )

    for smiles in data['SMILES_NEW']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fps.append({})
        else:
            fp = fp_generator.GetCountFingerprint(mol)
            fp_dict = dict(fp.GetNonzeroElements())
            fps.append(fp_dict)
    return fps


def vectorize_fingerprints(fps_dicts):
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(fps_dicts)
    feature_names = vec.get_feature_names_out()
    return pd.DataFrame(X, columns=feature_names), vec

def create_unhashed_fingerprint_dataframe(data, radius=2):
    fps_dicts = generate_unhashed_fingerprints(data, radius=radius)
    X_df, vec = vectorize_fingerprints(fps_dicts)

    # Join with Ames labels
    final_df = X_df.copy()
    final_df['AMES'] = data['AMES'].values
    return final_df, vec

def generate_unhashed_morgan_fingerprint_column(data, radius=2):
    smiles_list = data['SMILES_NEW'].tolist()
    fingerprints = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    fingerprints.append(np.nan)
                else:
                    # Unhashed fingerprint (sparse IntSparseIntVect)
                    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius)
                    fingerprints.append(fp)
            except ValueError:
                fingerprints.append(np.nan)

    data['UHFPS'] = fingerprints

def generate_morg_fingerprint_column(data,radius=2, fp_length=2048):
    smiles_list = data['SMILES_NEW'].tolist()
    fingerprints = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        fp_generator = GetMorganGenerator(radius=radius, fpSize=fp_length)
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Handle invalid SMILES gracefully
                    fingerprints.append(np.nan)
                else:
                    fp = fp_generator.GetFingerprint(mol)
                    fingerprints.append(fp)
            except ValueError:
                # Handle invalid SMILES gracefully
                fingerprints.append(np.nan)

    data['FPS'] = fingerprints

def fingerprints_to_bits(fp):
    return list(map(int, fp.ToBitString()))

def create_fingerprint_dataframe(data):
    # Convert fingerprints to lists of bits
    bit_lists = data['FPS'].apply(fingerprints_to_bits)

    # Create a DataFrame from the bit lists
    bit_df = pd.DataFrame(bit_lists.tolist(), index=data.index)

    # Add the "Expr" column
    bit_df['AMES'] = data['AMES']

    return bit_df

def main():
    # Loading training data
    ames_data = pd.read_csv("ames_mutagenicity_data_training.csv")

    # Adding molecule column
    PandasTools.AddMoleculeColumnToFrame(ames_data, smilesCol='SMILES_NEW', molCol='MOLECULE', includeFingerprints=False)
    print(ames_data)

    # Checking if there are any bad molecules
    next_mol_err = ames_data[ames_data['MOLECULE'].isna()].index.tolist()
    print(len(next_mol_err))

    # Making a hashed training dataframe with a fingerprint column and an AMES score column
    ames_data_h = ames_data.copy()
    generate_morg_fingerprint_column(ames_data_h)
    new_ames_data_h = create_fingerprint_dataframe(ames_data_h)
    new_ames_data_h.to_csv("ames_data_h_training.csv", index=False)
    print(new_ames_data_h)

    # Making an unhashed training dataframe with a fingerprint column and an AMES score column
    ames_data_uh = ames_data.copy()
    ames_data_uh_df, vectorizer = create_unhashed_fingerprint_dataframe(ames_data_uh)
    print(ames_data_uh_df)
    ames_data_uh_df.to_csv("ames_data_uh_training.csv", index=False)

    # Loading the validation dataset
    val_ames_data = pd.read_csv("ames_w_smiles_external.csv")

    # Making an unhashed validation dataset
    val_ames_data_df, vectorizer = create_unhashed_fingerprint_dataframe(val_ames_data)
    print(val_ames_data_df)
    val_ames_data_df.to_csv("ames_data_uh_validation.csv", index=False)

    # Making a hashed validation dataset
    generate_morg_fingerprint_column(val_ames_data)
    val_ames_data_h = create_fingerprint_dataframe(val_ames_data)
    val_ames_data_h.to_csv("ames_data_h_validation.csv", index=False)


if __name__ == "__main__":
    main()