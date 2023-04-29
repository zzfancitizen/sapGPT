import os
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from Util.token import OPENAI_API_KEY, ACTIVELOOP_TOKEN
import socket
import socks


def setup_proxy():
    proxy_host = '127.0.0.1'
    proxy_port = 30808
    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
    socket.socket = socks.socksocket


def build_docs(txt_dir):
    docs = []
    try:
        loader = TextLoader(txt_dir, encoding='utf-8')
        docs.extend(loader.load_and_split())
        return docs
    except Exception as e:
        pass


def text_spliter(docs):
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(docs)
    return texts


def save_to_deeplake(texts, embedding):
    username = 'zzfancitizen'
    db = DeepLake(dataset_path=f"hub://{username}/DummySet", embedding_function=embedding, public=False)
    db.add_documents(texts)


if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

    setup_proxy()
    embeddings = OpenAIEmbeddings(disallowed_special=())
    txt_dir = os.path.relpath('../DataSet/Data/output.txt')
    docs = build_docs(txt_dir)
    texts = text_spliter(docs)
    save_to_deeplake(texts, embeddings)
