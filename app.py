import streamlit as st
import pandas as pd
from backend import agents, rag, ocr
from config.settings import DISCLAIMER

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Health GenAI Demo", layout="wide")
st.title("🩺 Health GenAI — Demo (Mistral-powered OCR & RAG)")

# -----------------------------
# Sidebar: Mode Selection
# -----------------------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Chat", "Upload & RAG", "OCR", "About"])

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------
# 1️⃣ Chat Mode
# ----------------------------------------------------------------
if mode == "Chat":
    st.subheader("💬 Health Assistant Chat")

    # Display all chat messages
    for role, text in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)

    # Input box for user message
    user_input = st.chat_input("Ask a health-related question")
    if user_input:
        # Append user message
        st.session_state.history.append(("user", user_input))

        # Generate assistant response
        response = agents.smart_chat(user_input)
        st.session_state.history.append(("assistant", response))

        # Display only the new messages
        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response)

# ----------------------------------------------------------------
# 2️⃣ Upload & RAG Mode
# ----------------------------------------------------------------
elif mode == "Upload & RAG":
    st.subheader("📄 Upload Documents & Run RAG Search")

    uploaded_file = st.file_uploader(
        "Upload document (txt, pdf, docx, xlsx, png, jpg, jpeg)",
        type=["txt", "pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        with st.spinner("📥 Processing document..."):
            try:
                content = rag.extract_text_from_file(uploaded_file)
                rag.add_document(uploaded_file.name, content)
                st.success(f"✅ Added document: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to process file: {e}")

    query = st.text_input("🔍 Enter a query to search documents")

    if st.button("Search in documents", type="primary"):
        if not query.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("Searching knowledge base..."):
                results = rag.search(query, top_k=5)

            if results:
                st.write("### 📚 Top Matches:")
                shown = set()
                for doc_name, score, snippet in results:
                    if (doc_name, snippet) not in shown:
                        st.markdown(f"**📄 {doc_name}** (score: `{score:.2f}`)")
                        st.text_area("Snippet", snippet, height=120)
                        shown.add((doc_name, snippet))

                # Generate LLM answer from context
                st.markdown("### 🧠 Mistral-Powered Answer")
                answer = rag.answer_with_context(query, results)
                st.success(answer)
            else:
                st.info("No matches found in uploaded documents.")

    if st.button("📋 Show Uploaded Documents"):
        docs = rag.list_documents()
        if not docs:
            st.info("No documents added yet.")
        else:
            st.write("### 📑 Uploaded Documents:")
            for name in docs:
                st.markdown(f"- {name}")

# ----------------------------------------------------------------
# 3️⃣ OCR Mode
# ----------------------------------------------------------------
elif mode == "OCR":
    st.subheader("🖼 OCR — Extract & Structure Prescription")

    uploaded_img = st.file_uploader("Upload an image for OCR", type=["png", "jpg", "jpeg"])

    if uploaded_img:
        with st.spinner("🔍 Running OCR and structuring..."):
            result = ocr.ocr_stub(uploaded_img)

        st.markdown("### 📝 Extracted Text")
        st.text_area("OCR Text", value=result.get("raw_text", ""), height=200)

        st.markdown("### 🧾 Structured Prescription Data")
        st.json(result.get("parsed", {}))

        parsed = result.get("parsed", {})
        if isinstance(parsed, dict) and parsed.get("medicines"):
            df = pd.DataFrame(parsed["medicines"])
            st.markdown("#### 💊 Medicines Table")
            st.dataframe(df, use_container_width=True)

        st.markdown("### 🗣️ Prescription Summary")
        st.success(result.get("summary", "⚠️ Summary could not be generated."))

# ----------------------------------------------------------------
# 4️⃣ About Mode
# ----------------------------------------------------------------
else:
    st.header("ℹ️ About Health GenAI")
    st.markdown(DISCLAIMER)
    st.markdown("""
    **Health GenAI Demo** showcases:
    - 💬 Rule-based local chat  
    - 📄 Multi-file Retrieval-Augmented Generation (RAG) search  
    - 🖼 OCR-based prescription parsing structured via Mistral Inference API  
    """)
