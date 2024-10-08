from dotenv import load_dotenv, find_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import JSONNodeParser
import json
from flask import Flask, request, render_template

from llama_index.core.node_parser import SentenceSplitter


# Load environment variables
_ = load_dotenv(find_dotenv())
app=Flask(__name__)


def load_doc(file):
    documents = SimpleDirectoryReader(input_files=file).load_data()
    node_parser = SentenceSplitter(chunk_size=256)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    return nodes

def vectorstore(nodes):
    index = VectorStoreIndex.from_documents(nodes)
    return index

def index_retriever(index):
    retriever = index.as_retriever()
    return retriever


def query_engine(index):
    query_engine = index.as_query_engine()
    return query_engine