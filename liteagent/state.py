

class AgentState:
    def __init__(self,input_data=None):
        self.input_data = input_data or {}

    def get(self,key,default=None):
        return self.data_get(key,default)

    def set(self,key,value):
        self.input_data[key] = value

    def all(self):
        return self._data
    

    #Shares state between agents