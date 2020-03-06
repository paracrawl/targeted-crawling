import os
import configparser
import time
import pickle

def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)

######################################################################################
class Timer:
    def __init__(self):
        self.starts = {}
        self.cumm = {}

    def __del__(self):
        print("Timers:")
        for key, val in self.cumm.items():
            print(key, "\t", val)

    def Start(self, str):
        self.starts[str] = time.time()

    def Pause(self, str):
        now = time.time()
        then = self.starts[str]

        if str in self.cumm:
            self.cumm[str] += now - then
        else:
            self.cumm[str] = now - then

######################################################################################
class MySQL:
    def __init__(self, config_file):
        import mysql.connector
        config = configparser.ConfigParser()
        config.read(config_file)
        self.mydb = mysql.connector.connect(
        host=config["mysql"]["host"],
        user=config["mysql"]["user"],
        passwd=config["mysql"]["password"],
        database=config["mysql"]["database"],
        charset='utf8'
        )
        self.mydb.autocommit = False
        self.mycursor = self.mydb.cursor(buffered=True)

######################################################################################
def GetLanguages(configFile):
    filePath = 'pickled_domains/Languages'
    if not os.path.exists(filePath):
        print("mysql load Languages")
        sqlconn = MySQL(configFile)
        languages = Languages(sqlconn)
        with open(filePath, 'wb') as f:
            pickle.dump(languages, f)
    else:
        print("unpickle Languages")
        with open(filePath, 'rb') as f:
            languages = pickle.load(f)

    return languages

######################################################################################
class Languages:
    def __init__(self, sqlconn):
        self.coll = {}

        sql = "SELECT id, lang FROM language"
        sqlconn.mycursor.execute(sql)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            self.coll[res[1]] = res[0]
            self.maxLangId = res[0]
        
    def GetLang(self, str):
        str = StrNone(str)
        assert(str in self.coll)
        return self.coll[str]
        # print("GetLang", str)
