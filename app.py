import streamlit as st
import pandas as pd
import tempfile
import os
from fpdf import FPDF
import logging
from langchain_community.document_loaders import CSVLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Logging configuration
logging.basicConfig(level=logging.INFO)

def get_company_profile_text():
    """Formats company info and preferences into a single, structured string."""
    
    company_info = {
        "company_name": "LaTronic Solutions",
        "company_description": """
            LaTronic Solutions is a global IT services company established in 2008, with its headquarters in Northern Virginia.
            Over the past 15 years, the company has grown from a small technical support provider to a prominent player
            in the enterprise solutions space. Specializing in delivering innovative, scalable, and customized technology
            solutions, LaTronic helps businesses across various sectors achieve success through their cutting-edge services.
        """,
        "services": [
            "Program/Project Management",
            "Cloud Services (Microsoft Azure, AWS, Google Cloud)",
            "System Integration",
            "Infrastructure Solutions"
        ],
        "mission": "To empower clients with the tools and expertise needed to navigate the complexities of today’s fast-evolving digital landscape.",
        "leadership": {
            "background": "Extensive experience in the field, with contracts from US government and private sectors."
        }
    }

    company_preferences = {
        "expiration_date": ["3 months", "6 months", "9 months"],
        "preferred_areas": [
            "Program/Project Management",
            "Data Management and Analytics (AI/ML/LLM, Electronic Records Management, ETL)",
            "Cloud Infrastructure and Network Support (Microsoft Azure, AWS, Google Cloud)",
            "Application Development"
        ],
        "location": ["DMV Area", "Southeast Region of the US", "Outside CONUS"],
        "award_amount": ["$100K to $1M", "Less than $5M"],
        "naics_codes": ["541512 - IT-related services", "518210 - Cloud services", "541611 - Administrative Management"],
        "contract_type": ["IDIQ (BPA)", "MAC"]
    }
    
    # --- FIX: Pre-format the lists into strings first ---
    services_str = "- " + "\n- ".join(company_info['services'])
    areas_str = "- " + "\n- ".join(company_preferences['preferred_areas'])
    expiration_str = "- " + "\n- ".join(company_preferences['expiration_date'])
    locations_str = "- " + "\n- ".join(company_preferences['location'])
    awards_str = "- " + "\n- ".join(company_preferences['award_amount'])
    naics_str = "- " + "\n- ".join(company_preferences['naics_codes'])
    contracts_str = "- " + "\n- ".join(company_preferences['contract_type'])
    
    # Now the f-string only contains simple variables, no backslashes
    profile_text = f"""
# Company Profile: {company_info['company_name']}

## About Us
{company_info['company_description'].strip()}

### Our Mission
{company_info['mission']}

### Our Services
{services_str}

### Leadership Background
{company_info['leadership']['background']}

---

# Company Preferences for Opportunities

### Preferred Areas of Work
{areas_str}

### Target Contract Expiration
{expiration_str}

### Target Locations
{locations_str}

### Target Award Amount
{awards_str}
    
### Relevant NAICS Codes
{naics_str}

### Preferred Contract Types
{contracts_str}
"""
    return profile_text

try:
    os.environ["AZURE_OPENAI_API_KEY"] = "DVXL9D3Arjt5dHYXWIrDw0D71pTHgUuKbDSrArAnIvLRtIsP6lVsJQQJ99BFACYeBjFXJ3w3AAABACOG6tV6"  
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ls-openai-prd-01.openai.azure.com/"  
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"  
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "ls-gpt-4o-mini-06202025"  
    os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"] = "text-embedding-ada-002"  
except KeyError:
    st.error("❌ Critical credentials not found. Please check your environment variables.")
    st.stop()

# ---
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""
if "uploaded_file_object" not in st.session_state:
    st.session_state.uploaded_file_object = None

def on_file_upload():
    """Callback function to store the uploaded file object in session state."""
    st.session_state.uploaded_file_object = st.session_state.file_uploader_key

def set_prompt(question):
    """Callback function to update the prompt in session state."""
    st.session_state.prompt_input = question

# ---
def process_file_with_rag(file, file_type, user_prompt, output_format):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        if file_type == "xlsx":
            pd.read_excel(file).to_csv(tmp.name, index=False)
        else:
            tmp.seek(0)
            tmp.write(file.read())
        
        loader = CSVLoader(file_path=tmp.name)
        documents = loader.load()
    
    company_profile_doc = Document(page_content=get_company_profile_text(), metadata={"source": "Company Profile"})
    documents.append(company_profile_doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"])
    db = FAISS.from_documents(chunks, embeddings)

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], # Add this line
        temperature=0.5,
    )
    
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the user's question based only on the following context:
        <context>{context}</context>
        Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(db.as_retriever(), document_chain)
    
    response = retrieval_chain.invoke({"input": user_prompt})
    answer = response["answer"]
    
    output_filename = f"processed_output.{output_format.lower()}"
    if output_format == "CSV":
        pd.DataFrame([{"Prompt": user_prompt, "Answer": answer}]).to_csv(output_filename, index=False)
    elif output_format == "Excel":
        pd.DataFrame([{"Prompt": user_prompt, "Answer": answer}]).to_excel(output_filename, index=False)
    else: # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Prompt:\n{user_prompt}\n\nAnswer:\n{answer}")
        pdf.output(output_filename)
        
    return output_filename, answer

# ---
st.set_page_config(page_title="LaTronic Document Processor", layout="wide")
st.title("📁 LaTronic Solutions Document Processor")

# STEP 1: FILE UPLOAD
st.file_uploader(
    "**Step 1: Upload your data file (CSV or XLSX)**",
    type=["csv", "xlsx"],
    key="file_uploader_key",
    on_change=on_file_upload,
    help="Upload a file to activate the question and processing options."
)

# CONDITIONAL UI: SHOWS ONLY AFTER FILE UPLOAD
if st.session_state.uploaded_file_object is not None:
    uploaded_file = st.session_state.uploaded_file_object
    
    st.success(f"File ready for processing: **{uploaded_file.name}**")
    st.markdown("---")
    
    # STEP 2: DYNAMIC PROMPT CREATION
    st.text_area("**Step 2: Create your question below, or use the interactive helpers.**", key="prompt_input")

    service_options = [
        "Program/Project Management", "Data Management and Analytics", "AI/ML/LLM",
        "Electronic Records Management", "ETL", "Cloud Infrastructure and Network Support",
        "Microsoft Azure", "Amazon Web Services", "Google Cloud Platform",
        "Application Development", "Other Data-related"
    ]
    selected_service = st.selectbox(
        "**Optional: Choose a service to include in a sample question.**",
        options=service_options, index=5
    )

    sample_questions = [
        f"From `{uploaded_file.name}`, list all opportunities related to **{selected_service}**. The result should include the contract name, NACIS code (if available), description, and any other relevant details that indicate project management responsibilities",
        f"From `{uploaded_file.name}`, create a categorized report of all opportunities in our key areas (Data, Cloud, Project Management, etc.). For each, list the project name, NACIS code (if available), description, and expected outcomes."
    ]

    st.markdown("**Click a sample question to use it:**")
    cols = st.columns(len(sample_questions))

    for i, question in enumerate(sample_questions):
        with cols[i]:
            # The button now uses on_click and args instead of an if-statement block.
            st.button(
                question,
                key=f"sample_{i}",
                on_click=set_prompt,
                args=[question], # Pass the question text to the callback
                use_container_width=True
            )

    # STEP 3: FINAL PROCESSING
    st.markdown("**Step 3: Process the file with your final question.**")
    file_format = st.selectbox("Select output format", ["PDF", "CSV", "Excel"])
    
    if st.button("🚀 Process File", use_container_width=True, type="primary"):
        if not st.session_state.prompt_input:
            st.error("❌ Please enter a question in the text box first.")
        else:
            with st.spinner("🤖 Processing your file with AI... This may take a moment."):
                try:
                    file_type = uploaded_file.name.split(".")[-1]
                    output_path, result_answer = process_file_with_rag(
                        file=uploaded_file, file_type=file_type,
                        user_prompt=st.session_state.prompt_input, output_format=file_format
                    )
                    
                    st.success("✅ File processed successfully!")
                    with st.expander("📄 View Answer", expanded=True):
                        st.markdown(result_answer)

                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Result", data=f, file_name=output_path)
                
                except Exception as e:
                    logging.error("Error during processing", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a file to begin.")