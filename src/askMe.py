import os
import socket
import socks
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from Util import OPENAI_API_KEY, ACTIVELOOP_TOKEN
from langchain.memory import ConversationTokenBufferMemory, ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN


class AskMe:
    def __init__(self):
        self.__setup_proxy()

        # self.chat_history = []
        self.embeddings = OpenAIEmbeddings(disallowed_special=())
        self.username = 'zzfancitizen'
        self.db = DeepLake(dataset_path=f"hub://{self.username}/DummySet", read_only=True,
                           embedding_function=self.embeddings)
        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs['distance_measure'] = 'cosine'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['maximal_marginal_relevance'] = True
        self.retriever.search_kwargs['k'] = 10

        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        self.model = ChatOpenAI(model='gpt-3.5-turbo')
        self.qa = ConversationalRetrievalChain.from_llm(self.model, retriever=self.retriever, memory=self.memory,
                                                        chain_type='map_reduce')

    def __setup_proxy(self):
        self.proxy_host = '127.0.0.1'
        self.proxy_port = 30808
        socks.set_default_proxy(socks.SOCKS5, self.proxy_host, self.proxy_port)
        socket.socket = socks.socksocket

    def ask(self, query):
        result = self.qa({"question": query})
        return result['answer']
