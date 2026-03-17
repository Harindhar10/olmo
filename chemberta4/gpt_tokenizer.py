from deepchem.feat import Featurizer
from typing import List
from chemberta4.prompt_templates import PROMPT_TEMPLATES
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

    def __init__(self, tokenizer, task_name, task_type="single_task"):

        self.tokenizer = tokenizer
        self.task_name = task_name
        self.task_type = task_type

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
        # datapoints is a dataframe with columns: smiles + one or more label columns
        label_cols = datapoints.columns.drop('smiles')

        smiles = datapoints['smiles']
        smiles_with_prompts = self.formatting_prompts_func(smiles)

        encodings = []
        for i, text in enumerate(smiles_with_prompts):
            enc = self._featurize(text)
            if self.task_type == "multi_task":
                enc['labels'] = torch.tensor(
                    datapoints.iloc[i][label_cols].values.astype(float),
                    dtype=torch.float32,
                )
            else:
                enc['labels'] = torch.tensor(
                    datapoints.iloc[i][label_cols[0]], dtype=torch.long,
                )
            encodings.append(enc)
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

        prompt_template = PROMPT_TEMPLATES[self.task_name]

        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN
        texts = []
        for molecule in examples:
            text = prompt_template.format(smiles=molecule) + EOS_TOKEN
            texts.append(text)
        return texts
