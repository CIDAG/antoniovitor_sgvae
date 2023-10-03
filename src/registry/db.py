import os
from pymongo import MongoClient

class DB():
    client = None

    @staticmethod
    def get_database():
        connection_string = os.getenv('DB_CONNECTION')
        db_name = os.getenv('DB_NAME')

        if (not DB.client):
            try:

                DB.client = MongoClient(connection_string)
                DB.instance = DB.client[db_name]
                return DB.instance
            except:
                print('Can\'t connect to database')
    
    @staticmethod
    def get_collection(name: str):
        db = DB.get_database()
        if db is not None: return db[name]