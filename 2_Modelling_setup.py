import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from rdkit import Chem, RDLogger
import rdkit.Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.AllChem import GetMorganFingerprint
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

def generate_unhashed_count_fingerprint_column(data, radius=2):
    fps = []

    for smiles in data['SMILES_NEW']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fps.append({})
        else:
            fp = GetMorganFingerprint(mol, radius=radius, useCounts=True, useFeatures=False)
            fp_dict = dict(fp.GetNonzeroElements())
            fps.append(fp_dict)
    return fps


def create_uh_fps_dataframe(data, radius=2, vec_uh =None):
    # Vectorizer learns to store the number of counts in the fingerprint from the training data.
    # Using it this way allows the same vectorizer to be used on validation data
    # The same number/position of counts will be obtained from the training data - without this the model cannot read validation dataset

    uh_fps_dict = generate_unhashed_count_fingerprint_column(data, radius=radius)

    if vec_uh is None:
        vec_uh = DictVectorizer(sparse=False)
        x = vec_uh.fit_transform(uh_fps_dict)
    else:
        x = vec_uh.transform(uh_fps_dict)

    x_df = pd.DataFrame(x, columns=vec_uh.get_feature_names_out())
    list_feat = vec_uh.get_feature_names_out()
    print(list_feat)

    # Join with Ames labels
    final_uh_df = x_df.copy()
    final_uh_df['AMES'] = data['AMES'].values
    return final_uh_df, vec_uh


def generate_hashed_count_fingerprint_column(data,radius=2, fp_length=2048):
    hc_fps = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        fp_generator = GetMorganGenerator(
            radius=radius,
            fpSize=fp_length,
            countSimulation=True
        )

        for smiles in data['SMILES_NEW']:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                hc_fps.append(np.nan)
                print('turd')
            else:
                # Using a count fingerprint which is make into a dictionary.
                # This is different from the unhashed count fingerprint because it has a fixed length.
                fp = fp_generator.GetCountFingerprint(mol)
                fp_dict = dict(fp.GetNonzeroElements())
                hc_fps.append(fp_dict)

    return hc_fps


def create_hc_fps_dataframe(data, radius=2, vec_hc=None):
    # Adding a vector argument to this function allows the same vectorizer to be used for multiple datasets
    # This is essentially the same code as the previous vectorizer
    # It needs its own function in case the hashed count fingerprint has a different number of items that need to go into the vector
    hc_fps_dict = generate_hashed_count_fingerprint_column(data, radius=radius)

    if vec_hc is None:
        vec_hc = DictVectorizer(sparse=False)
        x_hc = vec_hc.fit_transform(hc_fps_dict)
    else:
        x_hc = vec_hc.transform(hc_fps_dict)

    x_hc_df = pd.DataFrame(x_hc, columns=vec_hc.get_feature_names_out())

    # Join with Ames labels
    final_hc_df = x_hc_df.copy()
    final_hc_df['AMES'] = data['AMES'].values

    return final_hc_df, vec_hc


# The next three functions are pretty much the standard ones provided in the example.
# They give the hashed, binary morgan fingerprint.
def generate_hashed_binary_fingerprint_column(data,radius=2, fp_length=2048):
    smiles_list = data['SMILES_NEW'].tolist()
    fingerprints = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        fp_generator = GetMorganGenerator(
            radius=radius,
            fpSize=fp_length
        )

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


def create_hb_fingerprint_dataframe(data):
    # Convert fingerprints to lists of bits
    bit_lists = data['FPS'].apply(fingerprints_to_bits)

    # Create a DataFrame from the bit lists
    bit_df = pd.DataFrame(bit_lists.tolist(), index=data.index)

    # Add the "Expr" column
    bit_df['AMES'] = data['AMES']

    return bit_df


def main():
    RDLogger.DisableLog('rdApp.*')

    # TRAINING data
    # Loading training data
    print('Let\'s arrange the training data!\n')
    training_ames_data = pd.read_csv("clean_training_data.csv")

    # UNHASHED + Count
    training_data_uh = training_ames_data.copy()
    training_data_uh_df, vec_uh = create_uh_fps_dataframe(training_data_uh)
    print('Behold! The unhashed + count training dataset:\n',training_data_uh_df)
    training_data_uh_df.to_csv("training_ames_data_uh.csv", index=False)
    training_uh_fps_list = training_data_uh_df.columns.tolist()

    # HASHED + Binary
    training_data_hb = training_ames_data.copy()
    generate_hashed_binary_fingerprint_column(training_data_hb)
    training_data_hb_df = create_hb_fingerprint_dataframe(training_data_hb)
    print('Behold! The hashed + binary training dataset:\n', training_data_hb_df)
    training_data_hb_df.to_csv("training_ames_data_hb.csv", index=False)

    # HASHED + Count
    training_data_hc = training_ames_data.copy()
    training_data_hc_df, vec_hc = create_hc_fps_dataframe(training_data_hc)
    print('Behold! The hashed + count training dataset:\n', training_data_hc_df)
    training_data_hc_df.to_csv("training_ames_data_hc.csv", index=False)


    # VALIDATION data
    # Loading the validation dataset
    print('Let\'s arrange the validation data!\n')
    validation_ames_data = pd.read_csv("clean_validation_data.csv")

    # UNHASHED + Count
    validation_data_uh = validation_ames_data.copy()
    validation_data_uh_df, _ = create_uh_fps_dataframe(validation_data_uh, vec_uh=vec_uh)
    print('Behold! The unhashed + count validation dataset:\n', validation_data_uh_df)
    validation_data_uh_df.to_csv("validation_ames_data_uh.csv", index=False)
    validation_uh_fps_list = validation_data_uh_df.columns.tolist()

    # Sanity check
    assert validation_uh_fps_list == training_uh_fps_list

    # HASHED + Binary
    validation_data_hb = validation_ames_data.copy()
    generate_hashed_binary_fingerprint_column(validation_data_hb)
    validation_data_hb_df = create_hb_fingerprint_dataframe(validation_data_hb)
    print('Behold! The hashed + binary validation dataset:\n', validation_data_hb_df)
    validation_data_hb_df.to_csv("validation_ames_data_hb.csv", index=False)

    # HASHED + Count
    validation_data_hc = validation_ames_data.copy()
    validation_data_hc_df, _ = create_hc_fps_dataframe(validation_data_hc, vec_hc=vec_hc)
    print('Behold! The hashed + count validation dataset:\n', validation_data_hc_df)
    validation_data_hc_df.to_csv("validation_ames_data_hc.csv", index=False)


if __name__ == "__main__":
    main()
