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
from io import BytesIO
import datetime
import ast

# Logging configuration
logging.basicConfig(level=logging.INFO)

# --- Centralized function for all NAICS data in a predefined order ---
def get_naics_data():
    """
    Provides a single source of truth for NAICS codes and their descriptions,
    in a predefined order. Returns a list of codes and a map of descriptions.
    """
    naics_definitions = {
        "541512": "Primary Computer Systems Design Services",
        "518210": "Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services",
        "541513": "Computer Facilities Management Services",
        "541519": "Other Computer Related Services",
        "541611": "Administrative Management and General Management Consulting Services",
        "541511": "Custom Computer Programming Services",
        "541690": "Other Scientific and Technical Consulting Services",
        "541618": "Other Management Consulting Services",
        "611420": "Computer Training",
        "541990": "All Other Professional, Scientific and Technical Services",
        "561311": "Employment Placement Agencies",
    }
    return list(naics_definitions.keys()), naics_definitions

def get_company_profile_text():
    """Formats company info and preferences into a single, structured string."""
    _, naics_map = get_naics_data()
    naics_list_for_profile = [f"{code} - {desc}" for code, desc in naics_map.items()]

    company_info = {
        "company_name": "LaTronic Solutions",
        "company_description": "Global IT services company established in 2008, specializing in innovative, scalable, and customized technology solutions.",
        "services": ["Program/Project Management", "Cloud Services", "System Integration", "Infrastructure Solutions"],
        "mission": "To empower clients with the tools and expertise needed to navigate the complexities of today’s fast-evolving digital landscape.",
        "leadership": {"background": "Extensive experience in the field, with contracts from US government and private sectors."}
    }

    
    company_preferences = {
        "preferred_locations": ["Washington DC", "Virginia", "Maryland"],
        "preferred_award_amount": "$100,000 to $1,000,000",
        "preferred_timeline": "Opportunities with expiration dates within the next 3 to 6 months.",
        "naics_codes": naics_list_for_profile
    }
    
    locations_str = "- " + "\n- ".join(company_preferences['preferred_locations'])
    naics_str = "- " + "\n- ".join(company_preferences['naics_codes'])
    
    profile_text = f"""
        # Company Profile: {company_info['company_name']}
        ## About Us
        {company_info['company_description']}
        ### Our Mission
        {company_info['mission']}
        ---
        # Company Preferences for Opportunities
        This section outlines our strategic priorities for identifying new opportunities.

        ### Preferred Locations
        {locations_str}

        ### Target Award Amount
        {company_preferences['preferred_award_amount']}

        ### Target Timeline
        {company_preferences['preferred_timeline']}

        ### Relevant NAICS Codes
        {naics_str}
        """
    return profile_text

def get_target_naics_codes():
    """Extracts just the numeric part of the NAICS codes for filtering."""
    target_codes, _ = get_naics_data()
    return target_codes

def filter_data_by_naics(uploaded_file, file_type):
    """Reads a file, filters it by NAICS codes, and returns the filtered DataFrame."""
    logging.info("Starting NAICS pre-filtering...")
    df = pd.read_excel(uploaded_file) if file_type == "xlsx" else pd.read_csv(uploaded_file)
    
    target_codes = get_target_naics_codes()
    possible_naics_columns = ['naics', 'naics code', 'naics_code']
    
    naics_col_found = next((col for col in df.columns if col.lower().strip() in possible_naics_columns), None)

    if not naics_col_found:
        logging.warning("No NAICS column found. Skipping filtering.")
        st.warning("⚠️ No column named 'NAICS' or 'NAICS Code' found. Processing the full file.")
        return df, None

    df[naics_col_found] = df[naics_col_found].astype(str)
    filtered_df = df[df[naics_col_found].str.strip().str.startswith(tuple(target_codes))].copy()
    
    return filtered_df, naics_col_found

try:
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_ENDPOINT"]
    os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["AZURE_API_VERSION"]
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = st.secrets["AZURE_LLM_DEPLOYMENT"]
    os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"] = st.secrets["AZURE_EMBEDDING_DEPLOYMENT"]
except KeyError as e:
    st.error(f"❌ A required secret is missing: {e}. Please configure your secrets in the app settings.")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
for key in ["prompt_input", "uploaded_file_object", "filtered_dataframe", "naics_summary"]:
    if key not in st.session_state:
        st.session_state[key] = None

def on_file_upload():
    """Callback to store file and reset dependent states."""
    st.session_state.uploaded_file_object = st.session_state.file_uploader_key
    st.session_state.filtered_dataframe = None
    st.session_state.naics_summary = None

def set_prompt(question):
    """Callback to update the prompt."""
    st.session_state.prompt_input = question

def run_naics_pre_filter():
    """Filters data and generates a summary in a fixed order, including zero counts."""
    uploaded_file = st.session_state.uploaded_file_object
    if not uploaded_file:
        return

    file_type = uploaded_file.name.split(".")[-1]
    uploaded_file.seek(0)
    
    filtered_df, naics_col = filter_data_by_naics(uploaded_file, file_type)
    st.session_state.filtered_dataframe = filtered_df
    
    ordered_codes, naics_map = get_naics_data()
    summary_lines = ["### NAICS Code Summary"]

    if naics_col:
        value_counts = filtered_df[naics_col].str.strip().str[:6].value_counts()
        summary_lines.append(f"**Total relevant opportunities found: {len(filtered_df)}**\n---")
        
        for code in ordered_codes:
            count = value_counts.get(code, 0)
            description = naics_map.get(code, "No description available")
            summary_lines.append(f"- **{code} - {description}**: {count} opportunities")
    else:
        summary_lines.append("⚠️ Could not perform NAICS analysis as no NAICS column was found.")
    
    st.session_state.naics_summary = "\n".join(summary_lines)

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
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], 
        temperature=0.5,
    )
    

    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert procurement analyst for LaTronic Solutions. Your goal is to identify and rank the top 5 most promising opportunities from the provided data that align with the company's strategic goals.

        **Company Profile Context (Primary Guide):**
        The provided context includes LaTronic Solutions' profile. Pay close attention to their preferred NAICS codes (e.g., 541512, 518210) and preferred areas of work (e.g., Program/Project Management, Cloud Infrastructure). These are the company's goals.

        **Data Context & User's Question:**
        The context also includes a list of procurement opportunities. The user's question acts as a mandatory filter on this data.

        **Your Task (Follow these steps precisely):**
        1.  **Filter:** First, apply the user's question to find all relevant opportunities.
        2.  **Analyze & Score:** From the filtered results, analyze each opportunity against the LaTronic Solutions company profile. An opportunity that aligns with multiple company goals (e.g., a preferred NAICS code AND a preferred area of work) is a higher-value target.
        3.  **Rank:** Select the **Top 5** highest-scoring opportunities. If fewer than 5 match, rank all that do.
        4.  **Explain & Format:** Present the result as a numbered list. For each of the top 5, provide a clear, concise explanation for its ranking. State exactly which company goals it aligns with. For example: "Ranks #1 because it falls under our primary NAICS code 541512 and is in our preferred 'Cloud Services' area, and a concise description on what is this opportunity about."

        ---
        **Context:**
        <context>{context}</context>

        ---
        **Question:**
        {input}"""
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

# --- UI LAYOUT ---
st.set_page_config(page_title="LaTronic Document Processor", layout="wide")
st.title("📁 LaTronic Solutions Business Development Processor")

st.info(
    """
    **Welcome! Here’s how to use this app:**

    **1. NAICS Code Filter**
    - Uploard your file.
    - If your file has a 'NAICS' column, you can use this section to pre-filter your data based on our company's preferred codes.
    - Click **"Analyze & Pre-filter"** to see a summary and download the smaller, filtered file.
    - You can then use this new file in the RAG Filter section below.

    **2. RAG Filter**
    - If your file has no NAICS code, or you've already filtered it, proceed here.
    - Upload your filtered file from **"NAICS Code Filter"** or your original file that has no NAICS code.
    - This AI-powered filter already knows our company's goals (preferred locations, award amounts, etc.).
    - Simply upload your file, click the sample question, and hit **"Process File"** to get a filtered list of the top opportunities.
    """
)

st.markdown("### NAICS Code Filter")
st.file_uploader("Upload the data file (CSV or XLSX)",
    type=["csv", "xlsx"],
    key="file_uploader_key",
    on_change=on_file_upload
)

if st.session_state.uploaded_file_object:
    st.success(f"File ready: **{st.session_state.uploaded_file_object.name}**")
    st.markdown("---")
    
    st.markdown("**Optional: Analyze data by NAICS codes before asking a question.**")
    st.button("📊 Analyze & Pre-filter by NAICS Codes", on_click=run_naics_pre_filter, use_container_width=True)

    if st.session_state.naics_summary:
        with st.expander("NAICS Pre-filter Summary", expanded=True):
            if st.session_state.filtered_dataframe is not None and not st.session_state.filtered_dataframe.empty:
                st.markdown("##### Download the Pre-filtered Data")
                
                # Convert dataframe to CSV in-memory
                csv_data = st.session_state.filtered_dataframe.to_csv(index=False).encode('utf-8')
                
                # Convert dataframe to Excel in-memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.filtered_dataframe.to_excel(writer, index=False, sheet_name='FilteredNAICS')
                excel_data = output.getvalue()

                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="⬇️ Download as CSV",
                        data=csv_data,
                        file_name=f"filtered_{st.session_state.uploaded_file_object.name.split('.')[0]}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                with col2:
                    st.download_button(
                        label="⬇️ Download as Excel",
                        data=excel_data,
                        file_name=f"filtered_{st.session_state.uploaded_file_object.name.split('.')[0]}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
            st.markdown(st.session_state.naics_summary)
    
    st.markdown("---")

# --- A NEW uploader for the RAG file ---
st.divider()

st.markdown("### RAG Filter")
rag_file = st.file_uploader(
    "Upload the data file for RAG analysis (CSV or XLSX)",
    type=["csv", "xlsx"],
    key="rag_file_uploader"
)

# --- The following steps only appear AFTER a file is uploaded ---
if rag_file is not None:
    
    # STEP 2: Ask a question about the uploaded file
    st.markdown("##### Step 1: Ask a Question")
    st.text_area(
        "Enter your question below, or use a sample question.",
        key="prompt_input"
    )

    # (Sample question logic)
    today = datetime.date.today()
    three_months_later = today + datetime.timedelta(days=90)
    today_str = today.strftime("%Y-%m-%d")
    future_date_str = three_months_later.strftime("%Y-%m-%d")

    sample_questions = [
        "Please analyze all available opportunities."
    ]
    st.markdown("**Click a sample question to use it:**")
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i]:
            st.button(
                question,
                key=f"sample_{i}",
                on_click=set_prompt,
                args=[question],
                use_container_width=True
            )

    # STEP 3: Process the file with the question
    st.markdown("##### Step 2: Get Your Answer")
    file_format = st.selectbox("Select output format", ["CSV", "Excel", "PDF"])

    if st.button("🚀 Process File", use_container_width=True, type="primary"):
        if not st.session_state.prompt_input:
            st.error("❌ Please enter a question in Step 1.")
        else:
            with st.spinner("🤖 Processing your file with AI..."):
                try:
                    file_type = rag_file.name.split('.')[-1]
                    output_path, result_df = process_file_with_rag(
                        file=rag_file,
                        file_type=file_type,
                        user_prompt=st.session_state.prompt_input,
                        output_format=file_format
                    )
                    st.success(f"✅ Analysis complete! Found {len(result_df)} matching rows.")
                    with st.expander("📄 View Filtered Data", expanded=True):
                        st.dataframe(result_df)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Result", data=f, file_name=output_path)
                except Exception as e:
                    logging.error("Error during RAG processing", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a file to begin.")
