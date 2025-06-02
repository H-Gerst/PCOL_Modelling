import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Mol, PandasTools, AllChem, Draw, rdMolDescriptors
import cirpy
import pandas as pd

def cas_to_smiles(df):
    cas_list = df['CAS'].tolist()
    bucket = []
    i = 0
    while i < len(cas_list):
        try:
            fuckit = cirpy.resolve(cas_list[i], 'smiles')
            bucket.append(fuckit)
            print(i/len(cas_list)*100)
        except Exception as e:
            print(i, 'well aren\'t you stooopid')
        i += 1
    print(bucket)
    df['SMILES_NEW'] = bucket
    df['SMILES_NEW'] = df['SMILES_NEW'].astype("str")
    return df

def ring_sys_count(df):
    mol_list = df['SMILES_NEW'].tolist()
    ring_bucket = []
    i = 0
    while i < len(mol_list):
        try:
            ring_col = cirpy.resolve(mol_list[i], 'ringsys_count')
            ring_bucket.append(ring_col)
            print(i / len(mol_list) * 100)
        except Exception as e:
            print(i, 'well aren\'t you stooopid')
            ring_bucket.append('error here')
        i += 1
    print(ring_bucket)
    df['RING_SYS'] = ring_bucket
    return df

def clean_and_check(data, kind):
    PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES_NEW', molCol='MOLECULE', includeFingerprints=False)
    err = data[data['MOLECULE'].isna()].index.tolist()
    print('There are this many bad molecules in the training data:', len(err))
    clean_data = data.dropna(subset=['MOLECULE'])
    clean_err = data[data['MOLECULE'].isna()].index.tolist()

    if len(clean_err) > len(err):
        print('... Something is still wrong')
    else:
        print('Nice! the', kind, 'data is okay')

    print(clean_data.columns.to_list())
    print('Here is the cleaned', kind, 'data: \n', clean_data)

    return clean_data

def main():
    # Ignoring 'WARNING: Not removing hydrogen without neighbours'
    RDLogger.DisableLog('rdApp.*')

    # TRAINING DATA
    # Loading data for model training
    training_data = pd.read_csv("ames_mutagenicity_data_training.csv")

    # Checking that everything is okay so far
    print('Here is the initial training data: \n', training_data)
    cleaned_training_data = clean_and_check(training_data, kind='training')

    # Saving to csv
    cleaned_training_data.to_csv("clean_training_data.csv", index=False)


    # VALIDATION DATA
    # Loading the validation dataset
    validation_ames_data = pd.read_excel("new_data.xlsx")

    # Checking that everything is as expected
    print('Here is the initial validation data: \n', validation_ames_data)

    # Adding a molecule column from smiles
    cas_to_smiles(validation_ames_data)
    cleaned_validation_data = clean_and_check(validation_ames_data, kind='validation')

    # Saving the dataframe as csv
    cleaned_validation_data.to_csv("clean_validation_data.csv", index=False)


if __name__ == "__main__":
    main()
