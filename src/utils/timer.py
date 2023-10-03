import time
import pandas as pd
import json
class timer():
    def __init__(self, logs_path, event, data: dict):
        self.event = event
        self.data = data
        
        self.logs_path = logs_path

        if not self.logs_path.exists():
            self.logs = pd.DataFrame(columns=['log', 'data'])
            self.logs.to_csv(logs_path, index=False)
        self.logs = pd.read_csv(logs_path, index_col='log')
         
    def __enter__(self):
        self.start = time.perf_counter()
        return self.start
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        count = time.perf_counter() - self.start
        self.log({'log': self.event, 'data': {'time': count, **self.data}})

    def log(self, data: list or dict):
        if type(data) is dict: data = [data]
        for i in data:
            self.logs.at[i['log'], 'data'] = json.dumps(i['data'])
        self.logs.to_csv(self.logs_path)