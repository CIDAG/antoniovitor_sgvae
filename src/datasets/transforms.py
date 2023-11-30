import grammar.old_grammar as grammar
import pandas as pd
import numpy as np
import time

class SGVAEDataFormatter(object):
    def __init__(self, input_field, output_field, properties: list):
        self.input_field = input_field
        self.output_field = output_field
        self.properties = properties

    def __call__(self, item: pd.DataFrame):
        properties = {prop: item[prop] for prop in self.properties}

        input_smiles = item[self.input_field]
        ohe_input = grammar.parse_smiles_list([input_smiles])[0]

        if(self.output_field == self.input_field):
            output_smiles = input_smiles
            ohe_output = ohe_input
        else:
            output_smiles = item[self.output_field].to_list()
            ohe_output = grammar.parse_smiles_list([output_smiles])[0]
        
        return {
            'input': ohe_input,
            'output': ohe_output,
            'properties': properties
        }
    

class TwoSmilesFormatter(object):
    def __init__(self, input_field, output_field, properties: list):
        self.input_field = input_field
        self.output_field = output_field
        self.properties = properties

    def __call__(self, items: pd.DataFrame):
        properties = {prop: items[prop] for prop in self.properties}

        smiles_0 = items['Cation']
        smiles_1 = items['Anion']
        ohes = grammar.parse_smiles_list([smiles_0, smiles_1])
        ohe_input = np.concatenate(ohes, axis=0)

        if(self.output_field == self.input_field):
            ohe_output = ohe_input
        else:
            smiles_0 = items['Cation']
            smiles_1 = items['Anion']
            ohes = grammar.parse_smiles_list([smiles_0, smiles_1])
            ohe_output = np.concatenate(ohes, axis=0)
        
        return {
            'input': ohe_input,
            'output': ohe_output,
            'properties': properties
        }