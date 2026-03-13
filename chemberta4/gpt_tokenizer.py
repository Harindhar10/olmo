from deepchem.feat import Featurizer
from typing import List
try:
    from transformers import BertTokenizerFast
except ModuleNotFoundError:
    raise ImportError(
        'Transformers must be installed for BertFeaturizer to be used!')
    pass
import torch

class GPTFeaturizer(Featurizer):
    """GPT Featurizer.

    GPT Featurizer.
    The GPT Featurizer is a wrapper class for HuggingFace's GPTNeoXTokenizerFast.
    This class intends to allow users to use the GPTTokenizer API while
    remaining inside the DeepChem ecosystem.

    Examples
    --------
    >>> from deepchem.feat import GPTFeaturizer
    >>> from transformers import GPTNeoXTokenizerFast
    >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("allenai/OLMo-7b-hf", do_lower_case=False)
    >>> featurizer = GPTFeaturizer(tokenizer)
    >>> feats = featurizer.featurize(['D L I P [MASK] L V T'])

    """

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def featurize(self, datapoints: str, **kwargs) -> List[List[int]]:
        """
        Calculate encoding using HuggingFace's GPT

        Parameters
        ----------
        datapoint: str
            Arbitrary string sequence to be tokenized.

        Returns
        -------
        encoding: List
            List containing three lists: the `input_ids`, 'token_type_ids', and `attention_mask`.
        """
        # datapoints is a dataframe with columns 'smiles' and label column specific to the moleculenet dataset 

        datapoints.columns = ['smiles','y']
        smiles = datapoints['smiles']
        labels = datapoints['y']
        smiles_with_prompts = self.formatting_prompts_func(smiles)

        encodings = []
        for i,text in enumerate(smiles_with_prompts):
            encodings.append(self._featurize(text))
            encodings[i]['labels'] = torch.Tensor(labels[i])
        return encodings


    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """
        Calculate encoding using HuggingFace's GPTNeoXTokenizerFast

        Parameters
        ----------
        datapoint: List
            Arbitrary string sequence to be tokenized.

        Returns
        -------
        encoding: List
            List containing three lists: the `input_ids`, 'token_type_ids', and `attention_mask`.
        """

        # the encoding is natively a dictionary with keys 'input_ids', 'token_type_ids', and 'attention_mask'
        encoding = self.tokenizer(datapoint,
                                    truncation=True,
                                    padding = 'max_length',
                                    max_length=128,
                                    return_tensors="pt",
                                           **kwargs).data
        #.data is to convert the output to a dictionary, so np.array() operation works as intended
        return encoding
    
    def formatting_prompts_func(self, examples):

        bbbp_prompt = """"### Question: Does this molecule cross the 
                            blood brain barrier?

                            ### Molecule:
                            {}

                            ### Answer:
                        """

        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN
        texts = []
        for molecule in examples:
            text = bbbp_prompt.format(molecule) + EOS_TOKEN
            texts.append(text)
        return texts

