
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

def kill_yourself(df):
    cas_list = df['CAS'].tolist()
    bucket = []
    i = 0
    while i < len(cas_list):
        try:
            fuckit = cirpy.resolve(cas_list[i], 'smiles')
            bucket.append(fuckit)
        except Exception as e:
            print(i)
            print('well aren\'t you stooopid')
        i += 1
    print(bucket)
    df['SMILES_NEW'] = bucket
    df['SMILES_NEW'] = df['SMILES_NEW'].astype("str")
    return df

def main():
    ames_data = pd.read_excel("New_data.xlsx")
    kill_yourself(ames_data)
    print(ames_data)

    PandasTools.AddMoleculeColumnToFrame(ames_data, smilesCol='SMILES_NEW', molCol='MOLECULE', includeFingerprints=False)
    next_mol_err = ames_data[ames_data['MOLECULE'].isna()].index.tolist()
    print(len(next_mol_err))

    ames_data = ames_data.dropna(subset=['MOLECULE'])
    next_mol_err = ames_data[ames_data['MOLECULE'].isna()].index.tolist()

    print(len(next_mol_err))

    ames_data.to_csv("ames_w_smiles_external.csv", index=False)

if __name__ == "__main__":
    main()