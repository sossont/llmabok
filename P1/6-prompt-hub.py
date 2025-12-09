import dotenv
dotenv.load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain import hub

# prompt = PromptTemplate.from_template("{country}의 수도는?")
# prompt.save("capital.json")
# prompt.save("capital.yaml")

# from langchain_core.prompts import load_prompt
# prompt = load_prompt("capital.json")
# prompt = load_prompt("capital.yaml")


prompt = hub.pull("rlm/rag-prompt")
print(prompt)

prompt_owner = "bosornd"
prompt_title = "rag-prompt"
hub.push(f"{prompt_owner}/{prompt_title}", prompt)
