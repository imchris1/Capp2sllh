import os
import base64
import warnings
import streamlit as st
from annotated_text import annotated_text
from streamlit_chat import message
from PyPDF2 import PdfMerger, PdfFileWriter, PdfFileReader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Suppress specific warnings about empty pages
warnings.filterwarnings("ignore", message="Warning: Empty content on page")

# Set page config
st.set_page_config(page_title="DEMO", layout='wide')

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #f5f5f5;
        }
        .css-1d391kg {
            background-color: #333333;
        }
        .css-ffhzg2 {
            background-color: #333333 !important;
            border: 2px solid #f5a623;
        }
        .css-1v3dy6w {
            color: #f5a623 !important;
        }
        .css-hxt7ib {
            color: #f5f5f5;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #f5a623;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .stSelectbox>div {
            background-color: #333333;
            color: #f5f5f5;
        }
        .stTextArea>textarea {
            background-color: #2c2c2c;
            color: #f5f5f5;
            border: 2px solid #f5a623;
        }
        .stTabs>div>div>div {
            background-color: #1e1e1e;
            border: 2px solid #f5a623;
            border-radius: 5px;
            color: #f5a623;
        }
        .stTabs>div>div>div>button {
            color: #f5a623 !important;
        }
        .stMarkdown {
            color: #f5f5f5;
        }
        .stExpander>div {
            background-color: #333333;
            border: 1px solid #f5a623;
        }
        .stExpanderHeader>div {
            color: #f5a623;
        }
    </style>
""", unsafe_allow_html=True)

# Create main tabs with new tabs added
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ASK", "Upload", "Get Keyword", "Relevant Documents", "Document Analytics", "Document Comparison"])

###############################################################################
# TAB 1: ASK
###############################################################################
with tab1:
    st.markdown(":orange[QUESTIONS or KEYWORDS.]")
    st.markdown("")
    llm = Ollama(model="dolphin3")
    # Continue your existing code for the ASK tab...

###############################################################################
# TAB 2: UPLOAD
###############################################################################
with tab2:
    st.markdown(":orange[UPLOAD PDF FILE]")
    st.markdown("")
    # Continue your existing code for the UPLOAD tab...

###############################################################################
# TAB 3: GET KEYWORD (SUMMARY & KEYWORDS)
###############################################################################
with tab3:
    st.markdown(":orange[UPLOAD FILE GET SUMMARY AND KEYWORDS]")
    st.markdown("")
    # Continue your existing code for the GET KEYWORD tab...

###############################################################################
# TAB 4: RELEVANT DOCUMENTS
###############################################################################
with tab4:
    st.markdown(":orange[Search Relevant Documents]")
    st.markdown("")
    # Continue your existing code for the RELEVANT DOCUMENTS tab...

###############################################################################
# TAB 5: DOCUMENT ANALYTICS
###############################################################################
with tab5:
    st.markdown(":orange[DOCUMENT ANALYTICS]")
    st.markdown("")
    
    # Upload a PDF for analysis
    uploaded_file = st.file_uploader("Upload a PDF file for document analytics", type="pdf")
    
    if uploaded_file:
        # Save file temporarily
        tmp_file_path = os.path.join("TEMP_FILEs", uploaded_file.name)
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the document for basic analytics (word count, page count)
        document = PyMuPDFLoader(tmp_file_path).load()
        total_pages = len(document)
        total_text = " ".join([doc.page_content for doc in document])
        word_count = len(total_text.split())

        st.write(f"Total Pages: {total_pages}")
        st.write(f"Word Count: {word_count}")

        # Optionally generate more advanced analytics like sentiment analysis, NER, etc.
        # Placeholder for further analytics - you can integrate libraries like spaCy or transformers here.
        st.write("Advanced analytics can be added here for NLP-based analysis.")

        # Clean up
        os.remove(tmp_file_path)

###############################################################################
# TAB 6: DOCUMENT COMPARISON
###############################################################################
with tab6:
    st.markdown(":orange[DOCUMENT COMPARISON]")
    st.markdown("")
    
    # Upload two PDFs for comparison
    uploaded_files = st.file_uploader("Upload two PDF files for comparison", type="pdf", accept_multiple_files=True)
    
    if len(uploaded_files) == 2:
        # Save the uploaded files temporarily
        tmp_file_path1 = os.path.join("TEMP_FILEs", uploaded_files[0].name)
        tmp_file_path2 = os.path.join("TEMP_FILEs", uploaded_files[1].name)
        
        for uploaded_file, tmp_file_path in zip(uploaded_files, [tmp_file_path1, tmp_file_path2]):
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Load the two PDFs
        document1 = PyMuPDFLoader(tmp_file_path1).load()
        document2 = PyMuPDFLoader(tmp_file_path2).load()

        # Extract text for comparison
        text1 = " ".join([doc.page_content for doc in document1])
        text2 = " ".join([doc.page_content for doc in document2])

        # Simple comparison based on word overlap (you can add more sophisticated comparisons)
        words1 = set(text1.split())
        words2 = set(text2.split())
        common_words = words1.intersection(words2)
        similarity_percentage = (len(common_words) / len(words1)) * 100

        st.write(f"Similarity based on word overlap: {similarity_percentage:.2f}%")

        # Clean up
        os.remove(tmp_file_path1)
        os.remove(tmp_file_path2)
