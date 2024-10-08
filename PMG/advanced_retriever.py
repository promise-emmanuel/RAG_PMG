from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings


embed_model = OpenAIEmbedding(model_name = os.environ.get("EMBED_MODEL"))



def load_documents(file):
    # Load environment variables
    load_dotenv(find_dotenv())  
    
    if embed_model is None:
        raise ValueError("EMBED_MODEL environment variable is not set!")

    # Load documents
    documents = SimpleDirectoryReader(input_files=[file]).load_data()

    # merge into a single large document rather than the one document per page


    document = Document(text="\n\n".join([doc.text for doc in documents]))


    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[4096, 2048, 512])
    nodes = node_parser.get_nodes_from_documents([document])
    return nodes, node_parser

def VectorStore(nodes, node_parser):
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    embed_model = OpenAIEmbedding(model_name = os.environ.get("EMBED_MODEL"))

    Settings.llm = llm
    Settings.embed_model=embed_model
    Settings.node_parser = node_parser


    if not os.path.exists("../VectorStore"):
        os.makedirs("../VectorStore")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        
        leaf_nodes = get_leaf_nodes(nodes)

        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context)

        automerging_index.storage_context.persist(persist_dir="../VectorStore")

    else:
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="../VectorStore"))

    return automerging_index


def auto_merging_engine(automerging_index):
    
    load_dotenv(find_dotenv())
    cohere_rerank = CohereRerank(
        api_key=os.environ["COHERE_API_KEY"], 
        top_n=2,
    )   
       
    automerging_retriever = automerging_index.as_retriever(similarity_top_k=6)

    retriever = AutoMergingRetriever(
        automerging_retriever,
        automerging_index.storage_context,
        verbose= True
    )

    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[cohere_rerank]
    )
    return auto_merging_engine
 