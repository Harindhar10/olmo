# The following is the function used to generate benchmarking splits in ChemBERTa3.

import deepchem as dc
import os
import argparse
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict

def generate_deepchem_splits(dataset_names: List, 
                             output_dir: str, 
                             clean_smiles: bool=True, 
                             max_smiles_len: int=200):
    """
    Generates scaffold splits for DeepChem datasets, optionally cleans them, and saves as CSVs.

    Parameters
    ----------
    dataset_names: List(str)
        List of dataset names.
    output_dir: str
        Folder to save the CSVs.
    clean_smiles: bool
        Whether to filter SMILES strings by max length.
    max_smiles_len: int
        Maximum allowed SMILES length if cleaning is enabled.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"deepchem_split_log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    logging.info("Starting DeepChem dataset splitting...")

    for dataset_name in dataset_names:
        try:
            logging.info(f"Processing dataset: {dataset_name}")
            load_fn = getattr(dc.molnet, f'load_{dataset_name}')

            task_names, (train_set, valid_set, test_set), transformers = load_fn(
                featurizer=dc.feat.DummyFeaturizer(),
                transformers=[],
                splitter='scaffold',
                reload=False
            )

            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            split_names = ['train', 'valid', 'test']
            datasets = [train_set, valid_set, test_set]

            for split_name, split_set in zip(split_names, datasets):
                smiles_list = split_set.X
                y = split_set.y

                if clean_smiles:
                    mask = pd.Series(smiles_list).str.len() <= max_smiles_len
                    removed_count = (~mask).sum()
                    logging.info(f"{dataset_name} {split_name}: removed {removed_count} rows (SMILES > {max_smiles_len})")
                    smiles_list = [s for i, s in enumerate(smiles_list) if mask.iloc[i]]
                    y = y[mask.to_numpy()]

                data = {'smiles': smiles_list}
                data.update({task_names[i]: y[:, i] for i in range(len(task_names))})
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(dataset_dir, f"{split_name}.csv"), index=False)
                logging.info(f"Saved {split_name} split to {dataset_dir}/{split_name}.csv")

        except AttributeError:
            logging.error(f"Dataset '{dataset_name}' not found in DeepChem.molnet. Skipping.")
        except Exception as e:
            logging.exception(f"Error while processing {dataset_name}: {str(e)}")

    logging.info("All datasets processed.")


datasets = ['sider'] #['bbbp','bace_classification','clintox','hiv','tox21','sider', 'clearance']
generate_deepchem_splits(dataset_names=datasets,
                                 output_dir='splits',
                                 clean_smiles=True,
                                 max_smiles_len=200)