import grammar.old_grammar as grammar
import pandas as pd
import time

class SGVAEDataFormatter(object):
    def __init__(self, input_field, output_field, properties: list):
        self.input_field = input_field
        self.output_field = output_field
        self.property_0 = properties[0]
        self.property_1 = properties[1]

    def __call__(self, items: pd.DataFrame):
        input_smiles = items[self.input_field].to_list()
        ohe_input = grammar.parse_smiles_list(input_smiles)

        if(self.output_field == self.input_field):
            output_smiles = input_smiles
            ohe_output = ohe_input
        else:
            output_smiles = items[self.output_field].to_list()
            ohe_output = grammar.parse_smiles_list(output_smiles)

        property_0 = items[self.property_0].to_numpy()
        property_1 = items[self.property_1].to_numpy()
        
        return ohe_input, ohe_output, property_0, property_1
    