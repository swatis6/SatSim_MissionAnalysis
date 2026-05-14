import yaml
from pathlib import Path

yamlFiles = ["adcs.yaml", "comms.yaml", "environment.yaml", "mission.yaml", "spacecraft.yaml"]

yamlPaths = [Path(__file__).parent / yamlFile for yamlFile in yamlFiles]

class Config:
    
    class ADCS:
        def __init__(self):
            self.__arr = {}
            with open(yamlPaths[0], "r") as file:
                config = yaml.safe_load(file)
                if config is not None:
                    for key, value in config.items():
                        self.__arr[key] = value
                file.close()
        
    class Comms:
        def __init__(self):
            self.__arr = {}
            with open(yamlPaths[1], "r") as file:
                config = yaml.safe_load(file)
                if config is not None:
                    for key, value in config.items():
                          self.__arr[key] = value
                file.close()
        
    class Environment:
        def __init__(self):
            self.__arr = {}
            with open(yamlPaths[2], "r") as file:
                config = yaml.safe_load(file)
                if config is not None:
                    for key, value in config.items():
                          self.__arr[key] = value
                file.close()
                
        
        def getDT(self):
            return self.__arr["dt"]
        
        def getDuration(self):
            return self.__arr["duration"]

    class Mission:
        def __init__(self):
            self.__arr = {}
            with open(yamlPaths[3], "r") as file:
                config = yaml.safe_load(file)
                if config is not None:
                    for key, value in config.items():
                         self.__arr[key] = value
                file.close()

        def getDT(self):
            return self.__arr["dt"]
        
        def getDuration(self):
            return self.__arr["duration"]
        
    class Spacecraft:
        def __init__(self):
            self.__arr = {}
            with open(yamlPaths[4], "r") as file:
                config = yaml.safe_load(file)
                if config is not None:
                    for key, value in config.items():
                          self.__arr[key] = value
                file.close()
        
        def getInitialState(self):
            return self.__arr["initial_state"].copy()
         