from abc import ABC, abstractclassmethod
import json
from dotenv import load_dotenv
# from .db import DB
import pandas as pd
from pathlib import Path

load_dotenv()

class Registry(ABC):
    @abstractclassmethod
    def register():
        pass

    def log(self, data):
        self.register('log', data)

    def error(self, data):
        self.register('error', data)

    def warn(self, data):
        self.register('warn', data)


class FileRegistry(Registry):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.file = open(path, 'a')
    
    def register(self, type: str, data):
        self.file.write(json.dumps({'type': type, **data}) + '\n')

class CSVRegistry(Registry):
    df: pd.DataFrame
    def __init__(self, path) -> None:
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(path)
        else:
            self.df = pd.DataFrame()
    
    def register(self, type: str, data):
        new_row = pd.DataFrame([{'type': type, **data}])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.path, index=False)

# class DBRegistry(Registry):
#     def __init__(self, collection_name: str) -> None:
#         super().__init__()
#         self.collection_name = collection_name
#         self.collection = DB.get_collection(collection_name)
            
    
#     def register(self, type: str, data):
#         if self.collection is not None:
#             try:
#                 self.collection.insert_one({'type': type, **data})
#             except:
#                 print('An error has ocurred while inserting log in database')


class CombineRegistry(Registry):
    def __init__(self, registries) -> None:
        super().__init__()
        self.registries = registries

    def register(self, type: str, data):
        for collection in self.registries:
            collection.register(type, data)