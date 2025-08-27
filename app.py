from flask import Flask, render_template, request, jsonify, session
import os, uuid, math, re
import PyPDF2
import replicate
from collections import Counter, defaultdict

# ========= CONFIG =========
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "sk-feae5bceb62049fcaa7db239bc30c566")  # <-- replace or set env var
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# You can switch to a stronger model if you have access on Replicate.
# Example options you might try: "meta/meta-llama-3-70b-instruct", "meta/llama-2-70b-chat"
REPLICATE_MODEL = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
# =========================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

# In-memory document store: { doc_id: {"chunks": [str, ...], "meta": {...}} }
DOC_STORE = {}

# ---------------- PDF handling ----------------
def extract_text_from_pdf(pdf_file):
    """Extracts *all* text from a PDF file object."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)

def chunk_text(text, chunk_size=3000, overlap=200):
    """Split long text into overlapping chunks by characters (simple + robust)."""
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == n:
            break
        i = j - overlap  # overlap for continuity
        if i < 0:
            i = 0
    return chunks

# ---------------- Lightweight TF-IDF retrieval ----------------
STOPWORDS = set("""
a an the and or of in on for from to with without by as at is are was were be been being have has had do does did this that these those it its
you your yours he she they them their we our ours i me my mine not but if then else when while so than too very can could should would may might
about into over under again further more most some any each other such only own same just also because up down out off above below between
""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]

def tokenize_filtered(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]

def build_tfidf_index(chunks):
    """
    Build simple TF-IDF-like data structures for cosine similarity.
    Returns (chunk_tf_idf_vectors, idf_map)
    """
    # term frequencies per chunk
    tfs = []
    df = defaultdict(int)

    tokenized_chunks = []
    for ch in chunks:
        toks = tokenize_filtered(ch)
        tokenized_chunks.append(toks)
        tf = Counter(toks)
        tfs.append(tf)
        for term in set(toks):
            df[term] += 1

    N = len(chunks)
    idf = {}
    for term, d in df.items():
        # smoothed idf
        idf[term] = math.log((N + 1) / (d + 1)) + 1.0

    # build weighted vectors
    vectors = []
    for tf in tfs:
        vec = {}
        for term, freq in tf.items():
            vec[term] = freq * idf.get(term, 0.0)
        vectors.append(vec)

    return vectors, idf

def cosine_sim(vec_a, vec_b):
    # dot
    dot = 0.0
    for k, v in vec_a.items():
        b = vec_b.get(k)
        if b:
            dot += v * b
    # norms
    na = math.sqrt(sum(v*v for v in vec_a.values()))
    nb = math.sqrt(sum(v*v for v in vec_b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def build_query_vector(query, idf):
    tf = Counter(tokenize_filtered(query))
    vec = {}
    for term, freq in tf.items():
        vec[term] = freq * idf.get(term, 0.0)
    return vec

def rank_chunks(chunks, vectors, idf, query, top_k=3):
    qv = build_query_vector(query, idf)
    scored = []
    for i, cv in enumerate(vectors):
        s = cosine_sim(cv, qv)
        scored.append((s, i))
    scored.sort(reverse=True)
    best = [idx for _, idx in scored[:top_k]]
    return best

# ---------------- Replicate call ----------------
def call_model(context, question):
    """
    Ask the model to answer strictly from the provided context.
    """
    prompt = (
        "You are a helpful assistant. Answer the question ONLY using the provided context.\n"
        "If the answer is not present in the context, say: \"I couldn't find this in the document.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    output = replicate.run(
        REPLICATE_MODEL,
        input={
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 1200,
            "repetition_penalty": 1
        },
    )
    return "".join(output)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    # Extract + chunk
    full_text = extract_text_from_pdf(file)
    if not full_text.strip():
        return jsonify({"success": False, "error": "No extractable text found in PDF"}), 400

    chunks = chunk_text(full_text, chunk_size=3000, overlap=200)

    # Build retrieval index (TF-IDF)
    vectors, idf = build_tfidf_index(chunks)

    # Store server-side (small cookie holds only the id)
    doc_id = str(uuid.uuid4())
    DOC_STORE[doc_id] = {
        "chunks": chunks,
        "vectors": vectors,
        "idf": idf,
        "meta": {"pages_estimate": full_text.count("\f") + 1, "num_chunks": len(chunks)}
    }
    session["doc_id"] = doc_id

    return jsonify({
        "success": True,
        "message": "File uploaded and indexed successfully",
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "preview": full_text[:1500]
    }), 200

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"success": False, "error": "No question provided"}), 400

    doc_id = session.get("doc_id")
    if not doc_id or doc_id not in DOC_STORE:
        return jsonify({"success": False, "error": "Please upload a PDF first"}), 400

    entry = DOC_STORE[doc_id]
    chunks = entry["chunks"]
    vectors = entry["vectors"]
    idf = entry["idf"]

    # Retrieve top chunks
    top_idx = rank_chunks(chunks, vectors, idf, question, top_k=3)

    # Build a concise context with chunk headers
    chosen = []
    for i in top_idx:
        chosen.append(f"[Chunk {i+1}]\n{chunks[i]}")
    context = "\n\n---\n\n".join(chosen)

    # Call the model with only the most relevant content
    try:
        answer = call_model(context, question)
    except Exception as e:
        return jsonify({"success": False, "error": f"Model call failed: {e}"}), 500

    return jsonify({
        "success": True,
        "answer": answer,
        "used_chunks": [i+1 for i in top_idx]  # 1-based indices for display
    }), 200

if __name__ == "__main__":
    # Important for VS Code debugger: avoid reloader extra process
    app.run(debug=True, use_reloader=False)
