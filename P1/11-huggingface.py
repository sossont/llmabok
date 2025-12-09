import dotenv
dotenv.load_dotenv()

# import os
# from huggingface_hub import login
# login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="rombodawg/Rombos-LLM-V2.5-Qwen-32b",
    task="text-generation",
)

response = llm.invoke("한국의 수도는?")
print(response)
