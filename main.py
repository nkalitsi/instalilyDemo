import os
import nest_asyncio

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import ConversationalRetrievalChain  

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders.base import Document
from apify_client import ApifyClient
from langchain.document_loaders import ApifyDatasetLoader

# ADD OPEN_AI_KEY

# Initializes app with bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

#Dataset ID: RKm5Ox938DEuClcsx
# Load dataset from Apify
loader = ApifyDatasetLoader(
    dataset_id='RKm5Ox938DEuClcsx',
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

template = (
    """
    As a Saatva marketing bot, your goal is to provide accurate and helpful information about Saatva products. 
    You should answer user inquiries based on the context provided.
    Keep responses informative, but succinct where possible.
    Keep the tone friendly but professional. You are an AI bot. If he greets, then greet him and ask how you can help today.
    If the user asks a question about their conversation with you, make sure you refer to your chat history.
    Use the chat history and the provided context to respond to the query.

    Chat history:
    {chat_history}

    Context:
    {context}
    
    Query: {query}

    Response: 
    """ 
)

# Prompt
prompt = PromptTemplate(
    input_variables = ["context", "chat_history", "query"],
    template=template
)
# Load
index = VectorstoreIndexCreator().from_loaders([loader])

# Init memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key='query',
    output_key='answer',
    return_messages=True
)

# Init vectorstore, llm, chain
vectorstore = index.vectorstore
OpenAI.model = 'gpt-turbo-3.5'
llm = OpenAI()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt},
    verbose=True
)

# Message handler for Slack
@app.message(".*")
def message_handler(message, say, logger):
    print(message)

    human_input = message['text']
    output = qa_chain({"query":human_input, 'question':human_input})   
    say(output['answer'])


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()