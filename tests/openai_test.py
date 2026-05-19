import os

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv("../project/.env")
assert os.getenv("OPENAI_BASE_URL")
assert os.getenv("OPENAI_API_KEY")

# client = OpenAI(
#     base_url = os.getenv("OPENAI_BASE_URL"),
#     api_key = os.getenv("OPENAI_API_KEY")
# )
# response = client.chat.completions.create(
#     model='gpt-4o-mini',
#     messages=[
#         {"role": "user", "content": "tell me a joke"}
#     ],
#     temperature=0
# )
# response = response.choices[0].message.content.strip()
# print(response)

client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
gpt_4o_mini = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(openai_client=client)
)
agent = Agent(gpt_4o_mini, name="Test agent")
response = agent.run_sync("Tell me a joke")
print(response)