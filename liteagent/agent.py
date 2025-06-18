from functools import wraps

class Agent:
    def __init__(self,name,func,model,prompt_template = None, tools = None):
        self.name = name
        self.func = func
        self.model = model
        self.prompt_template = prompt_template or None
        self.tools = tools or None