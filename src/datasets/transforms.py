import grammar.old_grammar as grammar
import pandas as pd
import time

class SGVAEDataFormatter(object):
    def __init__(self, input_field, output_field, properties: list):
        self.input_field = input_field
        self.output_field = output_field
        self.properties = properties

    def __call__(self, items: pd.DataFrame):
        # properties = {prop: items[prop].to_numpy() for prop in self.properties}

        # input_smiles = items[self.input_field].to_list()
        # ohe_input = grammar.parse_smiles_list(input_smiles)

        # if(self.output_field == self.input_field):
        #     output_smiles = input_smiles
        #     ohe_output = ohe_input
        # else:
        #     output_smiles = items[self.output_field].to_list()
        #     ohe_output = grammar.parse_smiles_list(output_smiles)
        
        # return {
        #     'input': ohe_input,
        #     'output': ohe_output,
        #     'properties': properties
        # }
        properties = {prop: items[prop] for prop in self.properties}

        input_smiles = items[self.input_field]
        ohe_input = grammar.parse_smiles_list([input_smiles])[0]

        if(self.output_field == self.input_field):
            output_smiles = input_smiles
            ohe_output = ohe_input
        else:
            output_smiles = items[self.output_field].to_list()
            ohe_output = grammar.parse_smiles_list([output_smiles])[0]
        
        return {
            'input': ohe_input,
            'output': ohe_output,
            'properties': properties
        }