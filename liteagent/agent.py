from functools import wraps

class Agent:
    def __init__(self,name,func,model,prompt_template = None, tools = None):
        self.name = name
        self.func = func
        self.model = model
        self.prompt_template = prompt_template 
        self.tools = tools or None

    def run(self,input_data,state):
        prompt = self.prompt_template.format(input_data,state)
        response = self.model.generate(prompt)
        return self.func(response,state)

def agent(name,model,prompt_template=None,tools=None):
    def decorator(func):
        a = Agent(name,func,model,prompt_template,tools)
        @wraps(func)
        def wrapper(input_data,state):
            return a.run(input_data,state)
        wrapper.agent = a
        return wrapper
    return decorator