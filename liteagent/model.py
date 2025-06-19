import openai
import anthropic

class LLMProvider:
    def __init__(self,model_name,api_key):
        self.model_name = model_name
        self.api_key = api_key

    