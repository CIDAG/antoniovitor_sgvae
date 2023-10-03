def select(dataset_name: str, subset:str = None):
    if(dataset_name == 'IL_ESW'):
        from datasets.IL_ESW import ESW_IL_Dataset
        return ESW_IL_Dataset(subset=subset)
    elif(dataset_name == 'IL_ESW_extended'):
        from datasets.IL_ESW_extended import ESW_IL_Extended_Dataset
        return ESW_IL_Extended_Dataset(subset=subset)
    elif(dataset_name == 'IL_ESW_extended_subset'):
        from datasets.IL_ESW_extended import ESW_IL_Extended_Subset
        return ESW_IL_Extended_Subset(subset=subset)

