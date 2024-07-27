from dotenv import load_dotenv, find_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import JSONNodeParser
import json
from flask import Flask, request, render_template



# Load environment variables
_ = load_dotenv(find_dotenv())
app=Flask(__name__)

documents = SimpleDirectoryReader(input_files=["data/Pmg_lds.md"]).load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@app.route("/", methods=["POST", 'GET'])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        response = query_engine.query(query)    
        return render_template('index.html', query=query, response=response)
    return render_template('index.html')

if __name__ =="__main__":
    port = int(os.environ.get('port', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)