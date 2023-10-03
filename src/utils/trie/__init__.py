class Trie():
    def __init__(self):
        self.root = dict()
    
    def set(self, iterator, data = None):
        current_dict = self.root
        for i in iterator: current_dict = current_dict.setdefault(i, {})

        current_dict['_exists'] = True
        if data is not None:
            current_dict['_data'] = data

    def has(self, iterator):
        current_dict = self.root
        for char in iterator:
            if char not in current_dict:
                return False
            
            current_dict = current_dict[char]

        if('_exists' not in current_dict):
            return False
        
        return current_dict['_exists']

    def get_data(self, iterator):
        current_dict = self.root
        for i in iterator:
            if i not in current_dict:
                return None
            
            current_dict = current_dict[i]
        
        return current_dict.get('_data')