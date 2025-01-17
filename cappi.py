import os
import base64
import warnings
import textwrap
import streamlit as st
import speech_recognition as sr
import subprocess
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from annotated_text import annotated_text
from streamlit_chat import message
# PDF management
from PyPDF2 import PdfMerger, PdfFileWriter, PdfFileReader
# LangChain components
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
# If you are using custom or community versions, keep these;
# otherwise replace them with official langchain imports:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyMuPDFLoader as CommunityPyMuPDFLoader
from langchain_community.vectorstores import FAISS as CommunityFAISS
warnings.filterwarnings("ignore", message="Warning: Empty content on page")
os.system('clear')
if not os.path.exists("DOCUMENTS"):
    os.makedirs("DOCUMENTS")
if not os.path.exists("TEMP_FILEs"):
    os.makedirs("TEMP_FILEs")
###############################################################################
#                                  CSAC
###############################################################################
class AdvancedDocumentAI:


    def __init__(self, file_path: str, save_path: str = "AIBRAIN"):

        # 1. Split PDF into text chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.loader = PyMuPDFLoader(file_path=file_path)
        chunks = self.text_splitter.split_documents(self.loader.load())


        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        self.vectorstore.save_local(save_path)

        # 4. Build a default summarization template (system prompt)
        #    You can also store advanced instructions, e.g. chain-of-thought or style guidelines.
        self.summarization_template = """
        ### System:
        You are an assistant tasked with two things:
        1. Summarize the provided document into one sentence, 9 words long.
        2. Generate a list of the top 10 search term keywords that are most relevant to the document's content.

        Ensure that the summary is concise and informative, while the keywords should reflect the most important topics or themes from the document.

        Your response should be clear, relevant, and limited to the information present in the document.
        !ONLY OUTPUT DATA NEVER SPEAK!

        ### Context:
        {context}

        ### User:
        Please summarize the document and provide 10 relevant keywords based on its content.

        ### Response:
        """

        # 5. Build the retrieval QA chain that uses the above summarization template
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OllamaLLM(model="dolphin3"),
            retriever=self.vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': PromptTemplate.from_template(self.summarization_template)
            }
        )

    def summarize_and_generate_keywords(self, pdf_path: str):
        """
        Summarize the given PDF and generate a list of keywords.

        Args:
            pdf_path (str): Path to the PDF document to be summarized/keyworded.

        Returns:
            tuple(str, List[str]): The summary (str) and a list of keywords (List[str]).
        """
        document = PyMuPDFLoader(pdf_path).load()
        full_text = " ".join([doc.page_content for doc in document])

        if not full_text.strip():
            return "BLANK", ["BLANK"] * 4

        # Run the chain
        response = self.qa_chain.invoke({'query': full_text})

        # Parse out the summary and keywords
        split_response = response['result'].split("Keywords:")
        summary = split_response[0].strip()
        keywords = split_response[1].strip() if len(split_response) > 1 else ""

        keywords_list = [keyword.strip() for keyword in keywords.split(',')]
        return summary, keywords_list
###############################################################################
#        !top of code!         STREAMLIT        !top of code!
###############################################################################
st.set_page_config(page_title="Chris's Ai PDF Parser", layout='wide')
st.title(":orange[C.A.P.P]")
st.subheader("We have AI ", divider=True)
###############################################################################
#                                   CSS
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
###############################################################################
tab1, tab2, tab3, tab4 ,tab5 ,tab6 ,tab7 ,tab8 ,tab9 ,tab10 ,tab11 = st.tabs(["Home", "UPLOAD", "KEYWORDS", "RELEVANT DOCUMENTS","-","INTELLIGENT DOCUMENT ANALYZER","CODING","CODE REVIEW","AI CHAT","RAG","*Playground*"])

def extract_snippets(text, keyword, window=50):
    """
    Extracts snippets of text around each occurrence of the keyword.
    :param text: The full text to search within.
    :param keyword: The keyword to search for.
    :param window: Number of characters to include before and after the keyword.
    :return: List of snippets containing the keyword.
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    snippets = []
    for match in pattern.finditer(text):
        start = max(match.start() - window, 0)
        end = min(match.end() + window, len(text))
        snippet = text[start:end].replace('\n', ' ')
        snippets.append(snippet)
    return snippets

def handle_tab9_submit():
    user_input = st.session_state.user_input_tab9
    
    if user_input.strip():
        # 1) Append the user's message to the session-state chat history
        st.session_state.chat_history_tab9.append({"role": "user", "content": user_input})

        # 2) Construct the prompt: system prompt + conversation history
        conversation_text = system_prompt
        for entry in st.session_state.chat_history_tab9:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                conversation_text += f"### User:\n{content}\n"
            else:
                conversation_text += f"### Assistant:\n{content}\n"

        # Final prompt for the assistant to respond
        conversation_text += "### Assistant:\n"

        # 3) Query the LLM
        with st.spinner("Thinking..."):
            try:
                llm_response = llm_tab9(conversation_text)
                # 4) Append the assistant response to chat history
                st.session_state.chat_history_tab9.append({"role": "assistant", "content": llm_response})
                # 5) Display the assistant response
                message(llm_response, is_user=False)
            except Exception as e:
                st.error(f"An error occurred: {e}")

        # 6) Clear the user input field
        st.session_state.user_input_tab9 = ""

def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return processor, model

def transcribe(audio):
    """
    Transcribes an audio input into text using the Whisper model.

    Parameters:
    - audio: A tuple of (sampling_rate, raw_audio)

    Returns:
    - transcription: The transcribed text from the audio input.
    """
    if audio is None:
        return "No audio input detected."

    sr, y = audio  # Unpack sampling rate and raw data

    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono

    y = y.astype(np.float32)  # Ensure data type is float32
    y /= np.max(np.abs(y))  # Normalize audio to [-1, 1]

    # Resample if needed (Whisper expects 16kHz audio)
    if sr != 16000:
        from scipy.signal import resample
        new_length = int(len(y) * (16000 / sr))
        y = resample(y, new_length)
        sr = 16000

    # Load Whisper processor and model
    processor, model = load_whisper_model()

    # Prepare the audio for Whisper
    input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription




with tab1:
    st.markdown(":green[_______________________________________________________]")
    

with tab2:
    st.markdown(":orange[UPLOAD PDF FILE]")
    st.markdown("")
    
    # You can customize the default target directory here
    target_dir = "DOCUMENTS"
    full_target_path = os.path.join("/mnt/g/AI/Projects/work/test", target_dir)
    if not os.path.exists(full_target_path):
        os.makedirs(full_target_path)

    uploaded_files = st.file_uploader(
        "Choose a PDF file to upload", accept_multiple_files=True, type=["pdf"]
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                bytes_data = uploaded_file.read()
                file_name = os.path.basename(uploaded_file.name)
                save_path = os.path.join(full_target_path, file_name)

                with open(save_path, "wb") as f:
                    f.write(bytes_data)

                st.success(f"File successfully uploaded: {save_path}")

            except Exception as e:
                st.error(f"An error occurred while uploading {uploaded_file.name}: {e}")

with tab3:
    st.markdown(":orange[UPLOAD FILE GET SUMMARY AND KEYWORDS]")
    st.markdown("")

    uploaded_files = st.file_uploader(
        "Choose PDF files to summarize", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Save temporarily
                tmp_file_path = os.path.join("TEMP_FILEs", uploaded_file.name)
                with open(tmp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the uploaded file using the advanced AI
                doc_ai = AdvancedDocumentAI(file_path=tmp_file_path, save_path="AIBRAIN")
                summary, keywords = doc_ai.summarize_and_generate_keywords(pdf_path=tmp_file_path)

                # Display results
                st.subheader(f":blue[Summary for {uploaded_file.name}]")
                st.write(summary)
                st.subheader(f":green[Keywords for {uploaded_file.name}]")
                st.write(", ".join(keywords))

                # Clean up
                os.remove(tmp_file_path)
                st.success(f"File {uploaded_file.name} processed and cleaned up.")

            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")

        # Optionally remove the TEMP_FILEs folder if empty
        if not os.listdir("TEMP_FILEs"):
            try:
                os.rmdir("TEMP_FILEs")
            except OSError as e:
                st.warning(f"Could not remove TEMP_FILEs folder: {e}")
    else:
        st.warning("Please upload one or more PDF files to summarize.")

with tab4:
    st.markdown(":orange[Search Relevant Documents]")
    st.markdown("")

    documents_folder = "./DOCUMENTS"
    if not os.path.exists(documents_folder):
        st.warning("Documents folder does not exist.")
    else:
        # List all files in the documents folder
        files = [
            f for f in os.listdir(documents_folder)
            if os.path.isfile(os.path.join(documents_folder, f)) and f.endswith('.pdf')
        ]

        if files:
            st.write("Here are all the PDF files in the Documents folder:")
            for file in files:
                st.write("- ", file)

            # Keyword input
            keyword = st.text_input("Enter a keyword to search for:")

            if keyword:
                st.write(f"Searching for the keyword '{keyword}'...")

                llm_for_search = Ollama(model="dolphin3")

                best_match_file = None
                best_match_score = -1

                # Iterate through each file and compute relevance
                for file in files:
                    file_path = os.path.join(documents_folder, file)
                    try:
                        loader = PyMuPDFLoader(file_path)
                        document = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=20
                        )
                        chunks = text_splitter.split_documents(document)

                        embedding_model = HuggingFaceEmbeddings(
                            model_name="all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        vectorstore = FAISS.from_documents(chunks, embedding_model)

                        # Build a retrieval chain for checking relevance
                        # You can tweak the System prompt to create more advanced scoring logic
                        template = """
                        ### System:
                        You are an assistant that helps identify the document most relevant to the user's keyword.
                        Only use data from the provided documents to determine relevance.
                        Provide a numeric score if possible, or a short text if not.
                        
                        ### Context:
                        {context}

                        ### User:
                        {question}

                        ### Response:
                        """

                        retrieval_chain = RetrievalQA.from_chain_type(
                            llm=llm_for_search,
                            retriever=vectorstore.as_retriever(),
                            chain_type="stuff",
                            chain_type_kwargs={'prompt': PromptTemplate.from_template(template)}
                        )

                        response = retrieval_chain.invoke({'query': keyword})

                        # Attempt to parse a numeric "relevance score"
                        # If the LLM returns text, we fallback to length-based approximation
                        try:
                            relevance_score = float(response['result'])
                        except ValueError:
                            relevance_score = len(response['result'])

                        st.write(f"File: {file} - Relevance Score: {relevance_score}")

                        if relevance_score > best_match_score:
                            best_match_score = relevance_score
                            best_match_file = file

                    except Exception as e:
                        st.error(f"Error processing file {file}: {e}")

                # Show best match
                if best_match_file:
                    st.subheader(f"Best match for the keyword '{keyword}' is:")
                    st.write(f"**{best_match_file}**")
                else:
                    st.warning(f"No document found related to the keyword '{keyword}'.")
        else:
            st.warning("No PDF files found in the Documents folder.")

with tab5:
    st.markdown(":rainbow[HAHAHAHAHAHAHA]")
    st.markdown("")

with tab6:
    st.markdown(":blue[**Intelligent Document Analyzer**]")
    st.markdown("")

    documents_folder = "./DOCUMENTS"

    # Check if the documents folder exists
    if not os.path.exists(documents_folder):
        st.warning("📂 **Documents** folder does not exist. Please ensure the folder is present.")
    else:
        # List all PDF files in the documents folder
        files = [
            f for f in os.listdir(documents_folder)
            if os.path.isfile(os.path.join(documents_folder, f)) and f.lower().endswith('.pdf')
        ]

        if files:
            st.markdown("### 📄 **Available PDF Documents:**")
            selected_file = st.selectbox("Select a document to analyze:", files)
            
            if selected_file:
                st.markdown(f"### 🔍 **Analyzing Document:** `{selected_file}`")
                file_path = os.path.join(documents_folder, selected_file)
                
                try:
                    loader = PyMuPDFLoader(file_path)
                    document = loader.load()
                    
                    # Combine all pages' content into a single string for analysis
                    full_text = " ".join([page.page_content for page in document])
                    
                    llm = Ollama(model="dolphin3")  # Initialize the LLM
                    
                    # Pass 1: Generate Summary
                    summary_prompt = PromptTemplate(
                        template="Provide a concise summary in just bullets of flowchart of the following document:\n\n{document}",
                        input_variables=["document"]
                    )
                    summary = llm(summary_prompt.format(document=full_text))
                    
                    # Pass 2: Extract Key Themes and Topics
                    themes_prompt = PromptTemplate(
                        template="Identify the main themes and topics discussed in the following summary:\n\n{summary}",
                        input_variables=["summary"]
                    )
                    themes = llm(themes_prompt.format(summary=summary))
                    
                    # Pass 3: Entity Recognition
                    entities_prompt = PromptTemplate(
                        template="Extract key entities (people, organizations, locations, dates) from the following summary:\n\n{summary}",
                        input_variables=["summary"]
                    )
                    entities = llm(entities_prompt.format(summary=summary))
                    
                    # Pass 4: Actionable Insights and Recommendations
                    insights_prompt = PromptTemplate(
                        template="Based on the following summary and identified themes, provide actionable insights and recommendations:\n\nSummary:\n{summary}\n\nThemes:\n{themes}",
                        input_variables=["summary", "themes"]
                    )
                    insights = llm(insights_prompt.format(summary=summary, themes=themes))
                    
                    # Display the results
                    st.markdown("---")
                    
                    with st.expander("📄 **1. Document Summary:**"):
                        st.write(summary)
                    
                    with st.expander("🎯 **2. Key Themes and Topics:**"):
                        st.write(themes)
                    
                    with st.expander("🏷️ **3. Extracted Entities:**"):
                        st.write(entities)
                    
                    with st.expander("💡 **4. Actionable Insights and Recommendations:**"):
                        st.write(insights)
                    
                    # Optionally, allow users to download the insights
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Insights as Text",
                        data=textwrap.dedent(f"""
                            Document: {selected_file}

                            1. Summary:
                            {summary}

                            2. Key Themes and Topics:
                            {themes}

                            3. Extracted Entities:
                            {entities}

                            4. Actionable Insights and Recommendations:
                            {insights}
                        """),
                        file_name=f"{os.path.splitext(selected_file)[0]}_Insights.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ **Error analyzing document `{selected_file}`:** {e}")
        else:
            st.warning("No PDF files found in the **Documents** folder.")

with tab7:
    st.markdown("## :orange[** AI CG TUI1 **]")
    st.write("StreamLIT")

    # -- Initialize session_state variables --
    if "code_attempts" not in st.session_state:
        st.session_state.code_attempts = [] 
    if "final_code" not in st.session_state:
        st.session_state.final_code = "" 
    if "final_error" not in st.session_state:
        st.session_state.final_error = None

    user_request = st.text_input("DATA ENTRY :")
    llm = Ollama(model="qwen2.5-coder:7b")
    if st.button("Generate and Run Code") and user_request.strip():
        st.session_state.code_attempts.clear()
        st.session_state.final_code = ""
        st.session_state.final_error = None

        max_attempts = 10
        current_error = None
        new_code = "" 

        for attempt_number in range(1, max_attempts + 1):
            if attempt_number == 1:
                # First attempt: brand-new code from user's request
                prompt = f"""
You are a Python coding assistant. (do not add ```python) 
Write a complete Python script that accomplishes the following request:
\"\"\"{user_request}\"\"\"

Output only the code (no extra text).
                """
            else:
                # Next attempts: fix the last code based on the error
                prompt = f"""
You wrote some Python code for the following request:
\"\"\"{user_request}\"\"\"

That code returned this error:
\"\"\"{current_error}\"\"\"

Please fix the code. Output only the corrected code, 
without extra commentary or ```python fences.
                """

            # Call 
            new_code = llm(prompt).strip()
            
            new_code = new_code.replace("```python", "").replace("```", "")

            # Store 
            attempt_entry = {
                "request": user_request,
                "attempt_number": attempt_number,
                "code": new_code,
                "error": ""
            }

            # Write 
            temp_filename = "temp_code.py"
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write(new_code)

            # Run 
            process = subprocess.run(
                ["python", temp_filename],
                capture_output=True,
                text=True
            )

            if process.returncode == 0:
                # Success
                st.success(f"Code ran successfully on attempt #{attempt_number}!")
                st.session_state.final_code = new_code
                st.session_state.final_error = None
                attempt_entry["error"] = "No error. Code ran successfully."
                st.session_state.code_attempts.append(attempt_entry)
                break
            else:
                # Failure - store error, keep trying
                current_error = process.stderr
                attempt_entry["error"] = current_error
                st.session_state.code_attempts.append(attempt_entry)

        else:
    
            st.warning("All attempts exhausted. The code still has errors.")
            st.session_state.final_code = new_code
            st.session_state.final_error = current_error

  
    st.markdown("---")
    st.markdown("## **Attempt History**")
    if st.session_state.code_attempts:
        for attempt in st.session_state.code_attempts:
            st.markdown(f"**Attempt #{attempt['attempt_number']}**")
            st.code(attempt["code"], language="python")
            if attempt["error"]:
                st.error(attempt["error"])

    # -- Display Final Code (if any) --
    if st.session_state.final_code:
        st.markdown("---")
        st.markdown("### **Final Code**")
        st.code(st.session_state.final_code, language="python")
        if st.session_state.final_error:
            st.error(f"Code still produced error:\n\n{st.session_state.final_error}")
        else:
            st.success("No errors. Final code is above.")


    st.markdown("---")
    st.markdown("## **Revise Final Code**")

    revision_request = st.text_input("Enter your revision request:")
    if st.button("Revise Code"):
        if not st.session_state.final_code:
            st.warning("No final code to revise yet.")
        else:
            # We'll take the last final_code and ask the LLM to revise it
            original_code = st.session_state.final_code
            prompt = f"""
We have this Python code:
\"\"\"{original_code}\"\"\"

The user wants to revise it with the following request:
\"\"\"{revision_request}\"\"\"

DO NOT REMOVE ANY MAJOR CODE JUST UPDATE AND UPGRADE.

Please produce the *revised code* in full, with no extra text or explanations.
Remove any ```python fences from the output.
            """
            revised_code = llm(prompt).strip()
            revised_code = revised_code.replace("```python", "").replace("```", "")


            temp_filename = "temp_code.py"
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write(revised_code)

            process = subprocess.run(
                ["python", temp_filename],
                capture_output=True,
                text=True
            )

            if process.returncode == 0:
                st.success("Revised code ran successfully!")
                st.session_state.final_code = revised_code
                st.session_state.final_error = None
            else:
                st.error("Revised code produced an error!")
                st.session_state.final_code = revised_code
                st.session_state.final_error = process.stderr

            
            st.markdown("---")
            st.markdown("### **Revised Final Code**")
            st.code(st.session_state.final_code, language="python")
            if st.session_state.final_error:
                st.error(f"Error:\n\n{st.session_state.final_error}")
            else:
                st.success("No errors. Revised code is above.")                

with tab8:
    st.markdown("## :blue[Code Review & Transformation]")
    st.write(
        "Paste or drop a large block of text/code below, then give instructions for **each** iteration."
    )

    # 1) Text area to paste large code block
    user_code = st.text_area(
        "Paste your code (or any text) here:",
        height=300,
        key="user_large_code_input"
    )

    # 2) Slider: how many iterations/refinements do we want?
    num_iterations = st.slider(
        "Number of refinement iterations (each iteration has its own instruction)",
        min_value=1,
        max_value=5,
        value=2,
        help="How many times we apply different instructions to refine the answer."
    )

    # 3) Create a text area for each iteration's instruction
    #    We store them in st.session_state so that changing the slider doesn’t reset them each time.
    if "iteration_instructions" not in st.session_state:
        st.session_state.iteration_instructions = [""] * num_iterations
    
    # If the slider changes, adjust the list size
    if len(st.session_state.iteration_instructions) != num_iterations:
        # Copy over existing instructions so user doesn't lose them
        old_instructions = st.session_state.iteration_instructions
        st.session_state.iteration_instructions = []
        for i in range(num_iterations):
            if i < len(old_instructions):
                st.session_state.iteration_instructions.append(old_instructions[i])
            else:
                st.session_state.iteration_instructions.append("")

    # Render the text areas for instructions dynamically
    for i in range(num_iterations):
        st.session_state.iteration_instructions[i] = st.text_area(
            f"Instruction for BOT {i+1}:",
            value=st.session_state.iteration_instructions[i],
            key=f"instruction_{i}"
        )

    # 4) Button to run the LLM loop
    if st.button("Submit for Multi-Iteration Transformation", key="code_review_button"):
        if not user_code.strip():
            st.warning("Please paste some code or text before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Create a custom prompt template that includes a {context} slot.
                    # We'll feed each previous iteration’s response into 'context'.
                    code_review_prompt = PromptTemplate(
                        template="""
                        ### System:
                        You are a helpful AI that can analyze or transform code based on user instructions.
                        The user has provided some context from previous iterations (if any):

                        {context}

                        The user has also provided the following code/text:

                        {code}

                        ### User Instruction:
                        {instruction}

                        ### Response:
                        """,
                        input_variables=["context", "code", "instruction"]
                    )

                    llm_for_code_review = OllamaLLM(model="qwen2.5-coder:7b")
                    
                    # We'll store debug data for each iteration
                    iteration_prompts = []
                    iteration_responses = []

                    # This string accumulates the "memory" from previous iterations
                    accumulated_context = ""

                    # 5) Loop over the number of iterations, each with its own instruction
                    for i in range(num_iterations):
                        current_instruction = st.session_state.iteration_instructions[i]

                        # Format the prompt with the current accumulated context
                        current_prompt = code_review_prompt.format(
                            context=accumulated_context,
                            code=user_code,
                            instruction=current_instruction
                        )
                        iteration_prompts.append(current_prompt)

                        # Call the LLM
                        response = llm_for_code_review(current_prompt)
                        iteration_responses.append(response)

                        # Update our context with this iteration’s response,
                        # so the next iteration “remembers” it
                        accumulated_context += f"\n### Iteration {i+1} Response:\n{response}\n"

                    # 6) Debug/Logs: Show each iteration’s prompt and response
                    st.markdown("## Debug / Iterations Log")
                    for i in range(num_iterations):
                        st.markdown(f"### Iteration {i+1} Prompt:")
                        st.write(iteration_prompts[i])
                        st.markdown(f"### Iteration {i+1} Response:")
                        st.write(iteration_responses[i])
                        st.markdown("---")

                    # 7) Final Answer: typically we show the last iteration’s response
                    st.markdown("### :green[Final Answer (After All Iterations)]")
                    st.write(iteration_responses[-1])

                except Exception as e:
                    st.error(f"Error during code review: {e}")

with tab9:
    st.markdown(":orange[ Chat ]")

    # Define your LLM
    global llm_tab9
    llm_tab9 = Ollama(model="llama2-uncensored:latest")

    # Define your preset system prompt
    global system_prompt
    system_prompt = """### System:
Your name is Luca you are a Helpful assistant.
Speak always as Luca.
Luca likes to joke.
Luca is sarcastic.
"""

    # Initialize chat history in session state
    if 'chat_history_tab9' not in st.session_state:
        st.session_state.chat_history_tab9 = []

    # Define the callback function for submitting messages
    def handle_tab9_submit():
        user_input = st.session_state.user_input_tab9

        if user_input.strip():
            # 1) Append user message to chat history
            st.session_state.chat_history_tab9.append({"role": "user", "content": user_input})

            # 2) Construct conversation text
            conversation_text = system_prompt
            for entry in st.session_state.chat_history_tab9:
                role = entry["role"]
                content = entry["content"]
                if role == "user":
                    conversation_text += f"### User:\n{content}\n"
                else:
                    conversation_text += f"### Assistant:\n{content}\n"
            conversation_text += "### Assistant:\n"

            # 3) Query the LLM
            with st.spinner("Thinking..."):
                try:
                    llm_response = llm_tab9(conversation_text)
                    # 4) Append assistant response to chat history
                    st.session_state.chat_history_tab9.append({"role": "assistant", "content": llm_response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            # 5) Clear input field
            st.session_state.user_input_tab9 = ""

    # Display existing chat with unique keys for each message
    for i, chat_entry in enumerate(st.session_state.chat_history_tab9):
        if chat_entry["role"] == "assistant":
            # Assistant message (left aligned)
            message(chat_entry["content"], is_user=False, key=f"assistant_{i}")
        else:
            # User message (right aligned)
            message(chat_entry["content"], is_user=True, key=f"user_{i}")

    # Text input for user messages
    st.text_input(
        "Your message:",
        key="user_input_tab9",
        on_change=handle_tab9_submit
    )

with tab10:
    st.markdown(":orange[QUESTIONS or KEYWORDS.]")
    st.markdown("")
    llm = Ollama(model="dolphin3")
    @st.cache_resource
    class AdvancedQueryAI:
        def __init__(self, file_path: str):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            loader = PyMuPDFLoader(file_path=file_path)
            chunks = text_splitter.split_documents(loader.load())
            embedding_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            vectorstore.save_local("TEMP")
            query_template = """
            ### System:
            you may speak about anything but when spoken to about a document make sure data is only from document.
            Take the user input keyword or question and find the info in the given document that is related.
            Only use data from the document provided.
            Make sure to check for relevance.
            Never add any info that is found outside of the file.
            Keep the tone simple and clear.
            If you do not know, say you cannot find it with a cute emoji to keep the tone light."

            ### Context:
            {context}

            ### User:
            {question}

            ### Response:
            """

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': PromptTemplate.from_template(query_template)
                }
            )

        def query(self, user_input: str) -> str:
            """
            Queries the loaded PDF with the provided user_input question/keywords.

            Args:
                user_input (str): The query or keywords to ask about.

            Returns:
                (str): The model's response.
            """
            try:
                response = self.qa_chain({'query': user_input})
                return response['result']
            except Exception as e:
                return f"Error during query: {e}"

    # Load PDFs from the `DOCUMENTS` directory
    pdf_dir = './DOCUMENTS'
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # Let user select a PDF
    selected_pdf = st.selectbox("Select a PDF:", pdf_files)

    # If a PDF is selected, load it (caching the retrieval chain)
    if selected_pdf:
        oracle = AdvancedQueryAI(os.path.join(pdf_dir, selected_pdf))
        st.session_state.selected_pdf = selected_pdf

        # Display the PDF viewer in an expander
        st.markdown(f"**Currently Viewing:** {selected_pdf}")
        with st.expander("View PDF", expanded=False):
            with open(os.path.join(pdf_dir, selected_pdf), "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                pdf_display = (
                    f'<embed src="data:application/pdf;base64,{base64_pdf}" '
                    f'width="800" height="800" type="application/pdf">'
                )
                st.markdown(pdf_display, unsafe_allow_html=True)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Text area for user query
    user_input = st.text_area("Ask any question or keyword.", key="user_input", label_visibility='visible')
    submit_button = st.button("Submit")

    # Handle user submission
    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if selected_pdf:
            with st.spinner(f'Looking for anything related inside {selected_pdf}...'):
                try:
                    # Query the chain
                    result = oracle.query(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                    message(result, is_user=False)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please select a PDF before submitting a query.")


