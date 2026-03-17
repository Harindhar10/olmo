PROMPT_TEMPLATES = {
    # Single-task classification
    "bbbp": """### Question: Does this molecule permeate the blood-brain barrier?

### Molecule:
{smiles}

### Answer:
""",
    "bace_classification": """### Question: Is this molecule a BACE-1 inhibitor?

### Molecule:
{smiles}

### Answer:
""",
    "hiv": """### Question: Does this molecule inhibit HIV replication?

### Molecule:
{smiles}

### Answer:
""",
    "clintox": """### Question: Is this molecule clinically toxic?

### Molecule:
{smiles}

### Answer:
""",
    # Multi-task classification
    "sider": """### Question: What side effects does this drug cause?

### Molecule:
{smiles}

### Answer:
""",
    "tox21": """### Question: Predict toxicity across multiple assays for this molecule.

### Molecule:
{smiles}

### Answer:
""",
    # Regression
    "clearance": """### Question: Predict intrinsic hepatic clearance from SMILES.

### Molecule:
{smiles}

### Answer:
""",
    "delaney": """### Question: Predict aqueous solubility from SMILES.

### Molecule:
{smiles}

### Answer:
""",
    "freesolv": """### Question: Predict hydration free energy from SMILES.

### Molecule:
{smiles}

### Answer:
""",
    "lipo": """### Question: Predict lipophilicity from SMILES.

### Molecule:
{smiles}

### Answer:
""",
    "bace_regression": """### Question: What is the BACE-1 inhibition strength (pIC50) of this molecule?

### Molecule:
{smiles}

### Answer:
""",
    # Generation / pretraining
    "zinc20": """SMILES: {smiles}""",
    "pubchem": """SMILES: {smiles}""",
}
