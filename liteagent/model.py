import openai
import anthropic

class LLMProvider:
    def __init__(self,model_name,api_key):
        self.model_name = model_name
        self.api_key = api_key

    def openai_generate_text(self,prompt):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=1000
        )
        return response.choices[0].text
    
    def anthropic_generate_text(self,prompt):
        response = anthropic.Anthropic(api_key=self.api_key).messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    
