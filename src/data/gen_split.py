import pandas as pd
import sys
import os

from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict

RANDOM_STATE = 0
NUM_FOLDS = 4
MODALITIES = ['CR', 'DX']


def main():
    if len(sys.argv) <= 1:
        print("Usage: python3 gen_split.py <path to corda.csv>")
        exit(1)

    df = pd.read_csv(sys.argv[1])
    df = df[df.modality.isin(MODALITIES)]
    print("Data shape:", df.shape)
    print(df.groupby("institution").count())
    print("Total patients:", len(df.groupby('patient_id')))

    institution_id = {
        'molinette': 0,
        'mauriziano': 1,
        'sanluigi': 2,
        'monzino': 3
    }
    df['institution_name'] = df.institution
    df.institution = df.institution.map(lambda v: institution_id[v])

    splits = []
    for institution in sorted(df.institution.unique()):
        sub_df = df[df.institution == institution]
        print(f"Institution: {institution}, shape:", sub_df.shape)

        folds = []
        skf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (train_idx, test_idx) in enumerate(skf.split(sub_df, sub_df.covid, groups=sub_df.patient_id)):
            # print(f"-- Institution {institution} fold {fold}")
            train_df = sub_df.iloc[train_idx]
            test_df = sub_df.iloc[test_idx]

            assert len(set(train_df.patient_id.values).intersection(set(test_df.patient_id.values))) == 0
            folds.append([train_df.patient_id.values, test_df.patient_id.values])

        splits.append(folds)
    

    # idx [fold] (train, test)
    transposed = {}


    for fold in range(NUM_FOLDS):
        fold_train_ids, fold_test_ids = [], []
        for institution in sorted(df.institution.unique()):
            train_ids, test_ids = splits[institution][fold]
            fold_train_ids.extend(train_ids)
            fold_test_ids.extend(test_ids)
        transposed[fold] = (fold_train_ids, fold_test_ids)
    
    for fold in range(NUM_FOLDS):
        train_ids, test_ids = transposed[fold]
        print(f"Fold {fold}:", len(train_ids), len(test_ids))
        
        train_df = df[df.patient_id.isin(train_ids)]
        test_df = df[df.patient_id.isin(test_ids)]
        assert len(set(train_df.patient_id.values).intersection(set(test_df.patient_id.values))) == 0
        
        train_df.to_csv(os.path.join(os.path.dirname(sys.argv[1]), f"fold{fold}_train.csv"), index=False)
        test_df.to_csv(os.path.join(os.path.dirname(sys.argv[1]), f"fold{fold}_test.csv"), index=False)

if __name__ == '__main__':
    main()