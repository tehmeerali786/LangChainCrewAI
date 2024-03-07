from dotenv import load_dotenv 
load_dotenv()

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatLiteLLM

chat = ChatLiteLLM(model="gemini/gemini-pro")

messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
print(chat(messages))



