from pymongo import MongoClient

class MongoDB(object):
    CONNECTION_STRING = "mongodb://root:example@localhost:27017"
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.client = MongoClient(cls.CONNECTION_STRING)
            cls.instance = super(MongoDB, cls).__new__(cls)
        return cls.instance
    
    def get_db(self, db_name = 'sgvae_data'):
        return self.client[db_name]