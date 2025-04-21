from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
# Load PDF documents for direct rule lookup
pdf_docs = load_pdf_file(data='Data/')

import re

# Direct rule lookup function
def get_rule(rule_id: str) -> str | None:
    """Return the text of the specified rule from the loaded PDF documents."""
    # Regex to capture rule text until the next rule or end
    pattern = rf'(Rule\s*{re.escape(rule_id)}\s*\.?[\s\S]*?)(?=\nRule\s*\d+(?:\.\d+)*|\Z)'
    # Combine all page contents
    combined = "\n".join([doc.page_content for doc in pdf_docs])
    match = re.search(pattern, combined, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


index_name = "criminalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)


from langchain_openai import ChatOpenAI
# Metrics & logging: record query latency, token usage, success/failure rates
import time, threading
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

# In-memory metrics storage
metrics = {
    'total_requests': 0,
    'total_latency': 0.0,
    'successes': 0,
    'failures': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0,
}
metrics_lock = threading.Lock()

class MetricsCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture token usage from LLM responses."""
    def __init__(self, metrics_store, lock):
        super().__init__()
        self.metrics = metrics_store
        self.lock = lock
        self._start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # Record LLM call start time if needed
        self._start_time = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        # Extract token usage if available
        usage = {}
        try:
            usage = response.llm_output.get("token_usage", {}) or {}
        except Exception:
            usage = {}
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        with self.lock:
            self.metrics['prompt_tokens'] += prompt
            self.metrics['completion_tokens'] += completion
            self.metrics['total_tokens'] += total

# Attach callback handler to LLM
metrics_handler = MetricsCallbackHandler(metrics, metrics_lock)
callback_manager = CallbackManager([metrics_handler])
llm = ChatOpenAI(model="gpt-4", temperature=0.0, max_tokens=1200, callback_manager=callback_manager)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Record request start metrics
    start_time = time.time()
    with metrics_lock:
        metrics['total_requests'] += 1
    # Get user query and log
    user_query = request.form.get("msg", "").strip()
    import logging
    logging.info(f"User query: {user_query}")
    # Check for direct rule lookup (e.g., "what is rule 14.1?")
    m = re.match(r'(?i)^\s*(?:what\s+is\s+)?rule\s*([\d.]+)\s*\??$', user_query)
    if m:
        rule_id = m.group(1)
        rule_text = get_rule(rule_id)
        if rule_text:
            elapsed = time.time() - start_time
            with metrics_lock:
                metrics['total_latency'] += elapsed
                metrics['successes'] += 1
            return rule_text
    # Fallback to RAG chain for other queries
    try:
        # Invoke the RAG chain
        result = rag_chain.invoke({"input": user_query})
        answer = result.get("answer", "")
        logging.info(f"Answer: {answer}")
        # Record successful completion
        elapsed = time.time() - start_time
        with metrics_lock:
            metrics['total_latency'] += elapsed
            metrics['successes'] += 1
        return answer
    except Exception:
        import logging
        logging.exception("Error in chat handler")
        # Record failure
        with metrics_lock:
            metrics['failures'] += 1
        return "Sorry, something went wrong. Please try again."




@app.route("/metrics")
def metrics_dashboard():
    # Render basic metrics dashboard
    with metrics_lock:
        total_requests = metrics['total_requests']
        successes = metrics['successes']
        failures = metrics['failures']
        avg_latency = (metrics['total_latency'] / total_requests) if total_requests > 0 else 0
        prompt_tokens = metrics['prompt_tokens']
        completion_tokens = metrics['completion_tokens']
        total_tokens = metrics['total_tokens']
        success_rate = (successes / total_requests * 100) if total_requests > 0 else 0
        failure_rate = (failures / total_requests * 100) if total_requests > 0 else 0
    return render_template(
        'metrics.html',
        total_requests=total_requests,
        successes=successes,
        failures=failures,
        avg_latency=round(avg_latency, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        success_rate=round(success_rate, 2),
        failure_rate=round(failure_rate, 2),
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
