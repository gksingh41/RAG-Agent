import streamlit as st
import os
from dotenv import load_dotenv

# --- AGNO IMPORTS ---
from agno.agent import Agent
from agno.models.google import Gemini
from agno.vectordb.lancedb import LanceDb
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.youtube import YouTubeKnowledgeBase

# --- SETUP ---
load_dotenv()
st.set_page_config(page_title="The AI Expert", layout="wide")

# ==========================================
# 1. DATA SOURCES (The "Curriculum")
# ==========================================
# 👇 PASTE YOUR YOUTUBE LINKS HERE INSIDE THE QUOTES
YOUTUBE_VIDEOS = [
    "https://www.youtube.com/watch?v=NLOBYtfdxuM", 
    "https://www.youtube.com/watch?v=1j__7fVv9Ew"
]

# The folder where you put your PDFs
KNOWLEDGE_FOLDER = "knowledge" 

# ==========================================
# 2. INFRASTRUCTURE (The Brain & Memory)
# ==========================================
# We use a Local Embedder (Free, runs on your CPU)
embedder = SentenceTransformerEmbedder(
    id="sentence-transformers/all-MiniLM-L6-v2",
    dimensions=384,
)

# We use LanceDB (Local File Database)
vector_db = LanceDb(
    table_name="expert_knowledge", 
    uri="tmp/lancedb_expert", 
    embedder=embedder,
)

# ==========================================
# 3. INGESTION ENGINE (The "Training" Function)
# ==========================================
def initialize_brain():
    """Reads all PDFs and Videos and saves them to the Vector DB."""
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # --- PHASE 1: YOUTUBE ---
    if YOUTUBE_VIDEOS:
        status_container.info(f"📺 Processing {len(YOUTUBE_VIDEOS)} YouTube videos...")
        try:
            yt_kb = YouTubeKnowledgeBase(urls=YOUTUBE_VIDEOS, vector_db=vector_db)
            yt_kb.load(recreate=False) # Add to DB, don't delete old stuff
        except Exception as e:
            st.warning(f"Could not load some videos (might lack captions): {e}")
    
    progress_bar.progress(50)

    # --- PHASE 2: LOCAL FILES (PDF/TXT) ---
    if os.path.exists(KNOWLEDGE_FOLDER):
        files = os.listdir(KNOWLEDGE_FOLDER)
        status_container.info(f"📚 Reading {len(files)} local files from '{KNOWLEDGE_FOLDER}'...")
        
        for i, filename in enumerate(files):
            file_path = os.path.join(KNOWLEDGE_FOLDER, filename)
            
            # Handle PDFs
            if filename.lower().endswith(".pdf"):
                kb = PDFKnowledgeBase(path=file_path, vector_db=vector_db)
                kb.load(recreate=False)
                
            # Handle Text Files
            elif filename.lower().endswith(".txt"):
                kb = TextKnowledgeBase(path=file_path, vector_db=vector_db)
                kb.load(recreate=False)
                
    progress_bar.progress(100)
    status_container.success("✅ Brain Update Complete! I have learned everything.")

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🧠 Knowledge Base")
st.sidebar.caption("Manage what the AI knows.")

if st.sidebar.button("🚀 Re-Train AI"):
    with st.spinner("Ingesting data... (This may take a minute)"):
        initialize_brain()

st.sidebar.divider()
st.sidebar.markdown(f"**Sources Loaded:**")
st.sidebar.markdown(f"- 📹 {len(YOUTUBE_VIDEOS)} Videos Configured")
if os.path.exists(KNOWLEDGE_FOLDER):
    st.sidebar.markdown(f"- 📄 {len(os.listdir(KNOWLEDGE_FOLDER))} Files in Folder")

# ==========================================
# 5. THE AGENT (The Application)
# ==========================================
# PM Concept: "Unified Search"
# We connect the agent to the DB. It doesn't matter which 'KnowledgeBase' class 
# we use here (PDF or Text), as long as it points to the same 'vector_db'.
agent = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    knowledge=TextKnowledgeBase(path="knowledge/ai_guide.txt", vector_db=vector_db), 
    search_knowledge=True, # Force RAG
    show_tool_calls=True,  # Show "Thinking" process
    markdown=True,
)

# ==========================================
# 6. USER INTERFACE
# ==========================================
st.title("🎓 The AI Expert")
st.caption("Ask me about the PDFs and Videos you provided.")

query = st.text_input("Ask a question:")
    
if query:
    response = agent.run(query)
    st.write(response.content)
    
    # --- FIX: SAFE SOURCE CHECKING ---
    # We use 'getattr' to check if sources exist before trying to loop through them.
    # This prevents the crash if the agent didn't find any sources.
    sources = getattr(response, "sources", [])
    
    if sources:
        with st.expander("🔍 View Sources (Debug Info)"):
            for source in sources:
                st.write(source)
    else:
        st.caption("Note: No specific document source was cited for this answer.")