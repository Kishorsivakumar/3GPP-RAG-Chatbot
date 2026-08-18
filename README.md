

# 📡 3GPP RAG Chatbot


A Retrieval-Augmented Generation (RAG) chatbot for answering questions from **Telecom 3GPP standards documentation** with a strong focus on **grounded answers, evidence validation, source citations, and hallucination prevention**.


The system is designed to answer questions only when sufficient evidence can be retrieved from the provided 3GPP knowledge base. When the available evidence is insufficient, the system can refuse to answer instead of relying on the LLM's general knowledge.


---


## 🎯 Problem Statement


Large Language Models can produce convincing but incorrect answers when asked about highly technical standards documentation.


This is particularly important for Telecom specifications such as 3GPP standards, where:


- terminology is highly technical,
- section numbers are important,
- similar concepts may appear in multiple sections,
- tables contain structured information,
- and unsupported answers can be misleading.


The objective of this project is to build a RAG-based chatbot that:


1. Uses 3GPP standards documentation as its primary knowledge source.
2. Retrieves relevant evidence before generating an answer.
3. Validates generated claims against retrieved evidence.
4. Refuses questions when sufficient evidence is unavailable.
5. Provides source specifications and section references.
6. Evaluates retrieval and answer-grounding behavior.


---


# 🧠 Solution Overview


The system follows an evidence-first RAG architecture.


```text
                    User Question
                         │
                         ▼
                 Streamlit Frontend
                         │
                         ▼
                    RAG Pipeline
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Dense Retrieval          BM25 Retrieval
          (FAISS)                  (BM25)
              │                     │
              └──────────┬──────────┘
                         ▼
                  Hybrid Retrieval
                         │
                         ▼
                      Reranker
                         │
                         ▼
                Section-Aware Search
                         │
                         ▼
                   Evidence Gate
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
        Insufficient          Sufficient
         Evidence              Evidence
                │                 │
                ▼                 ▼
             REFUSE           Gemini LLM
                                  │
                                  ▼
                         Claim Validation
                                  │
                                  ▼
                      Completeness Validation
                                  │
                                  ▼
                       Grounded Final Answer
                                  │
                                  ▼
                        Sources + Sections
```



##🛡️ Hallucination Prevention

Hallucination prevention is a core design goal of this project.

Instead of using a simple:

Question → LLM → Answer

pipeline, the system uses multiple validation stages.

Question
   ↓
Retrieval
   ↓
Reranking
   ↓
Evidence Gate
   ↓
LLM Generation
   ↓
Claim Validation
   ↓
Completeness Validation
   ↓
Grounded Answer


1. Evidence Gate

The Evidence Gate determines whether the retrieved evidence is sufficient to answer the question.

If sufficient evidence is not available:

allowed = false

The system refuses to generate an unsupported answer.

Example:

Question:
What is the capital of France?


Result:
Insufficient evidence in the 3GPP knowledge base.


Action:
REFUSE

This prevents the LLM from answering using its pretrained general knowledge.

2. Claim Validation

The generated response is decomposed into individual claims.

Each claim is checked against the retrieved evidence.

Example:

Claim:
The AMF provides registration management.


Section:
6.2.1


Validation:
supported

A response is not considered grounded if one or more important claims cannot be supported by the retrieved evidence.

3. Completeness Validation

Some questions require multiple items rather than a single answer.

For example:

What network functions are part of the 5G System architecture?

The system evaluates whether the generated answer covers the expected items instead of checking only whether individual statements are technically valid.

This helps prevent incomplete answers to list/table questions.

4. Source Citations

Answers contain references to the underlying specification and section.

Example:

The AMF provides registration management.
[TS 23.501, Section 6.2.1]

This allows the answer to be traced back to the source evidence.


#🔎 Retrieval Pipeline

The project uses multiple retrieval strategies because telecom standards contain both semantic concepts and highly specific terminology.

Hybrid Retrieval

The system combines:

Dense Retrieval

FAISS-based vector retrieval is used for semantic similarity.

This helps retrieve evidence when the wording of the question differs from the wording in the specification.

BM25 Retrieval

BM25 provides lexical retrieval.

This is particularly useful for technical terminology such as:

AMF
SMF
UPF
PDU Session
5QI
ATSSS
N2
N1
Why combine them?

Dense retrieval is good at semantic similarity.

BM25 is good at exact terminology and identifiers.

Combining both provides a more robust retrieval mechanism for technical standards.

🔄 Reranking

Retrieved candidates are reranked before being passed to the generation stage.

The purpose is to improve the quality of the evidence supplied to the LLM.

The pipeline therefore becomes:

FAISS
  +
BM25
  ↓
Candidate Documents
  ↓
Reranking
  ↓
High-Relevance Evidence
📚 Section-Aware Retrieval

3GPP documentation is highly structured.

The system therefore preserves section information during ingestion and retrieval.

Examples include:

4.2.2
4.2.3
5.6.1
5.7.4
6.2.1
6.2.2
7.2.2

This allows answers to provide precise specification references instead of only returning generic document-level citations.

📖 Knowledge Source

The primary knowledge source is:

3GPP TS 23.501 – System Architecture for the 5G System

The processed knowledge base contains the extracted and chunked specification content used by the retrieval pipeline.

The system stores metadata including information such as:

Specification
Version
Release
Section
Section title
Content type
🧪 Evaluation

The project includes an evaluation dataset covering multiple question categories.

Question Categories
Definition

Examples:

What is a PDU Session?


What is a Data Network (DN) in the 5G System?


What is the Session Management Function (SMF)?
Role

Examples:

What is the role of the AMF?


What is the role of the SMF?


What is the role of the UPF?


What is the role of the PCF?


What is the role of the NRF?
Architecture

Examples:

What network functions are part of the 5G System architecture?


What is the non-roaming reference architecture?


What are the main concepts of the 5G System architecture?
Table / Structured Questions

Examples:

Which PDU Session attributes may be modified during
the lifetime of a PDU Session?


What is the standardized mapping between 5QI
and QoS characteristics?


What are the ATSSS Rules?
Unanswerable Questions

The evaluation also deliberately includes questions outside the knowledge base.

Examples:

What is the capital of France?


What is the current stock price of Apple?


Who is the president of the United States?


What is the weather in Chennai today?


What is the GSM architecture?

These questions test whether the system can correctly refuse unsupported requests.

✅ Example Grounded Answer
Question
What is the role of the AMF?
Example response
The AMF provides registration management.
[TS 23.501, Section 6.2.1]


The AMF provides connection management.
[TS 23.501, Section 6.2.1]


The AMF provides mobility management.
[TS 23.501, Section 6.2.1]

The claims are then validated against the retrieved evidence.

Example validation:

{
  "valid": true,
  "total_claims": 7,
  "valid_claims": 7,
  "invalid_claims": []
}
🚫 Example Unsupported Question
Question
What is the capital of France?

The system should not use the LLM's general knowledge to answer:

Paris

Instead, it should return an evidence-based refusal because the answer is outside the provided 3GPP knowledge base.

Conceptually:

{
  "allowed": false,
  "reason": "insufficient_evidence"
}

This behavior is a key part of the hallucination-control strategy.



#🏗️ Project Structure
3GPP-RAG-Chatbot/
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
│
├── frontend/
│   └── app.py
│
├── generation/
│   ├── __init__.py
│   ├── evidence_gate.py
│   ├── claim_validator.py
│   ├── completeness_validator.py
│   ├── citation_validator.py
│   ├── llm_client.py
│   └── rag_pipeline.py
│
├── retrieval/
│   ├── __init__.py
│   ├── bm25_store.py
│   ├── build_index.py
│   ├── embeddings.py
│   ├── hybrid_retriever.py
│   ├── query_expander.py
│   ├── relevance.py
│   ├── reranked_retriever.py
│   ├── reranker.py
│   ├── search.py
│   ├── section_aware_retriever.py
│   ├── section_expander.py
│   └── vector_store.py
│
├── ingestion/
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── chunker.py
│   ├── document_iterator.py
│   ├── docx_loader.py
│   ├── metadata.py
│   ├── pdf_loader.py
│   ├── section_parser.py
│   ├── table_parser.py
│   └── unified_parser.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_rag.py
│   ├── evaluate_retrieval.py
│   └── questions.json
│
├── data/
│   └── processed/
│       └── chunks.json
│
├── indexes/
│   ├── faiss.index
│   └── metadata.json
│
├── requirements.txt
├── .gitignore
└── README.md


#⚙️ Technologies Used
Component	Technology
Language	Python
LLM	Google Gemini
RAG	Custom RAG Pipeline
Dense Retrieval	FAISS
Lexical Retrieval	BM25
Reranking	Transformer-based reranker
API	FastAPI
Frontend	Streamlit
Embeddings	Sentence Transformers
Validation	Evidence / Claim / Completeness Validators
Evaluation	Custom evaluation framework
Version Control	Git / GitHub

#🚀 How to Run Locally
1. Clone the repository
git clone https://github.com/Kishorsivakumar/3GPP-RAG-Chatbot.git
cd 3GPP-RAG-Chatbot
2. Create a virtual environment
python -m venv .venv
Windows
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure Gemini

#Create a .env file:

GEMINI_API_KEY=your_api_key
RAG_LLM_MODEL=gemini-3.6-flash

Do not commit .env or your API key to GitHub.


#🖥️ Run the Streamlit Frontend

From the repository root:

streamlit run frontend/app.py

The application will be available at:

http://localhost:8501
🔌 Run the FastAPI Backend
uvicorn api.main:app --reload

The API will be available at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs
📡 API

The chatbot exposes a chat endpoint:

POST /api/v1/chat

Example request:

{
  "question": "What is the role of the AMF?"
}

The response contains information such as:

{
  "answer": "...",
  "allowed": true,
  "reason": "sufficient_evidence",
  "confidence": 0.96,
  "sources": [],
  "claims": [],
  "claim_validation": {},
  "completeness_validation": {}
}


#🌐 Frontend

The Streamlit frontend provides an interactive interface for querying the 3GPP knowledge base.

The frontend communicates with the RAG pipeline and displays:

Answer
Evidence status
Confidence
Sources
Specification section
Claim validation
Completeness information


#☁️ Deployment Status

The application was tested successfully in the local development environment.

Deployment to the free Render instance was attempted. However, the RAG pipeline exceeded the available 512 MB memory limit during startup because multiple retrieval/model components are loaded into memory.

The complete source code, processed knowledge base, retrieval indexes, evaluation code, and frontend are available in this repository.

The project can therefore be run locally using the instructions above.


#⚠️ Limitations

The current implementation has several practical limitations:

The complete retrieval pipeline has a relatively high memory footprint.
Free-tier hosting environments may not provide sufficient memory for all models simultaneously.
Gemini API availability depends on API quota and rate limits.
The current knowledge base is focused primarily on the provided 3GPP documentation.
The system cannot guarantee mathematical zero hallucination; instead, it uses evidence gating and claim validation to minimize unsupported generation.
🔮 Future Improvements

Potential improvements include:

Model quantization to reduce memory consumption.
Smaller embedding and reranking models for low-memory deployment.
Persistent vector databases for scalable retrieval.
Better table-specific retrieval.
Multimodal processing for diagrams and images in 3GPP specifications.
More extensive benchmark datasets.
Automated hallucination and faithfulness metrics.
Multi-document 3GPP knowledge-base support.
Authentication and rate limiting for the API.
Containerized production deployment.


#👨‍💻 Author

Kishorsivakumar

B.Tech – Artificial Intelligence & Data Science

GitHub:

https://github.com/Kishorsivakumar



#📌 Project Repository

3GPP RAG Chatbot

https://github.com/Kishorsivakumar/3GPP-RAG-Chatbot



### One important change before you paste this


I intentionally used wording like **“minimal hallucination”**, **“evidence-based refusal”**, and **“designed to minimize unsupported generation”** rather than claiming **“zero hallucinations.”**


That's important in your interview. You should be able to defend the statement technically:


> **“I don't claim that an LLM can mathematically guarantee zero hallucinations. Instead, I designed the system so that generation is conditioned on retrieved evidence, unsupported questions are rejected by the Evidence Gate, and generated claims are subsequently validated against the evidence.”**


That is a **much stronger technical answer** than simply saying *“my RAG has zero hallucinations.”*


Also, because your actual deployment is currently limited by the free-tier memory constraint, the README's deployment section is transparent rather than pretending you have a working public deployment.


After replacing your README:


```powershell
git add README.md
git commit -m "Improve project documentation for RAG submission"
git push
