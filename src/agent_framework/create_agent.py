# import os
# import asyncio
# from agent_framework import Agent
# from azure.identity.aio import AzureCliCredential
# from azure.ai.projects.aio import AIProjectClient
# from agent_framework.openai import OpenAIChatClient
# from dotenv import load_dotenv

# load_dotenv()

# # Load LLM
# client= OpenAIChatClient(
#     model    = "gpt-5-mini",  
#     api_key  = os.getenv("OPENAI_API_KEY"),
#     base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
# )



# async def main():

#     async with AzureCliCredential() as credential:
#         # Create agent in Azure Foundry
#         async with AIProjectClient(
#             endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#             credential = credential
#         ) as project_client:
            
#             created_agent = await project_client.agents.create_agent(
#                 model = "gpt-5-mini",
#                 name = "AssistantBot",
#                 instructions = "You are an assistant that explains query clearly"
#                              )

#             print(f"Agent Created Successfully ")
#             print(f"Agent ID : {created_agent.id}")

#             async with Agent (
#                     chat_client= AzureAIAgentClient(
#                     project_client = project_client,
#                     agent_id = created_agent.id )            
#                 ) as agent:
                                
                
#                 # model    = "gpt-5-mini",  
#                 # api_key  = os.getenv("OPENAI_API_KEY"),
#                 # base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
#                 #                )




#                 async for chunk in agent.run("Explain briefly about Microsoft Agent Framework", stream = True):
#                     if chunk.text:
#                       print(chunk.text, end = "", flush = True)

#                       print()
    
# if __name__ == "__main__":
#     asyncio.run(main())