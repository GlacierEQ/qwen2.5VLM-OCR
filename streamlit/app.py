import streamlit as st
import requests
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import markdown2
import pynvml

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; padding: 30px; }
    .stButton>button { 
        background-color: #28a745; 
        color: white; 
        border-radius: 8px; 
        padding: 8px 16px; 
        font-size: 16px; 
        margin-top: 10px; 
    }
    .stTextArea>label { 
        font-weight: bold; 
        color: #1a3c34; 
        font-size: 16px; 
    }
    .stImage>img { 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        margin-bottom: 15px; 
    }
    .sidebar .sidebar-content { 
        background-color: #ffffff; 
        padding: 20px; 
        border-right: 1px solid #e0e0e0; 
    }
    h1, h2, h3 { 
        color: #1a3c34; 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        margin-bottom: 20px; 
    }
    .stProgress .st-bo { 
        background-color: #28a745; 
    }
    .debug-expander { 
        background-color: #e6f3fa; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 15px; 
    }
    .markdown-preview { 
        background-color: #ffffff; 
        padding: 20px; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        min-height: 400px; 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        font-size: 16px; 
    }
    .stContainer { 
        margin-bottom: 20px; 
        padding: 15px; 
        border-radius: 8px; 
        background-color: #ffffff; 
    }
    </style>
""", unsafe_allow_html=True)

# vLLM API endpoint
VLLM_URL = "http://qwen-vlm:8000/v1/chat/completions"

# Dynamic MAX_WORKERS based on GPU memory
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free / 1024**3  # GB
        MAX_WORKERS = min(4, max(1, int(free_mem // 3)))  # ~3GB per request
    else:
        MAX_WORKERS = 2  # Fallback
    pynvml.nvmlShutdown()
except Exception:
    MAX_WORKERS = 2  # Fallback if pynvml fails

# Initialize session state
if "file_results" not in st.session_state:
    st.session_state.file_results = {}
if "file_times" not in st.session_state:
    st.session_state.file_times = {}
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

st.title("📄 Qwen2.5-VL OCR on PDFs")

# Sidebar for file selection and status
with st.sidebar:
    st.header("Uploaded Files")
    uploaded_files = st.file_uploader("Upload PDFs (max 200MB per file)", type=["pdf"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        st.write(f"Files uploaded: {len(uploaded_files)}")
    status_placeholder = st.empty()

def process_page(file_name, page, page_idx):
    """Process a single page, returning OCR result and timing."""
    page_start_time = time.time()
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract all text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }
    try:
        response = requests.post(VLLM_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
        else:
            text = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        text = f"Exception: {str(e)}"
    page_time = time.time() - page_start_time
    return file_name, page_idx, text, page_time, img_b64, page

# Process files only if new uploads or not yet processed
if uploaded_files and not st.session_state.processed:
    with status_placeholder.container():
        st.info("Processing PDFs in parallel...")
        progress_bar = st.progress(0)
        progress_text = st.empty()
    
    # Store uploaded file names to detect changes
    uploaded_file_names = [f.name for f in uploaded_files]
    if uploaded_file_names != [f["name"] for f in st.session_state.uploaded_files]:
        st.session_state.uploaded_files = [{"name": f.name, "data": f.read()} for f in uploaded_files]
        for f in uploaded_files:
            f.seek(0)  # Reset file pointers
    
    # Collect all pages
    all_pages = []
    total_pages = 0
    st.session_state.file_results = {}
    st.session_state.file_times = {}
    st.session_state.debug_info = {}
    
    for uploaded_file in st.session_state.uploaded_files:
        file_name = uploaded_file["name"]
        file_start_time = time.time()
        try:
            pages = convert_from_bytes(uploaded_file["data"], dpi=150)
            all_pages.extend([(file_name, page, i+1) for i, page in enumerate(pages)])
            total_pages += len(pages)
            st.session_state.file_times[file_name] = file_start_time
            st.session_state.file_results[file_name] = []
            st.session_state.debug_info[file_name] = {
                "pdf_size": len(uploaded_file["data"]),
                "page_count": len(pages)
            }
        except Exception as e:
            st.error(f"Failed to process {file_name}: {str(e)}")
    
    if all_pages:
        with status_placeholder.container():
            progress_text.text(f"Processing {total_pages} pages with {MAX_WORKERS} concurrent requests...")
        processed_pages = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {executor.submit(process_page, file_name, page, page_idx): (file_name, page_idx) for file_name, page, page_idx in all_pages}
            for future in as_completed(future_to_page):
                file_name, page_idx = future_to_page[future]
                try:
                    file_name, page_idx, text, page_time, img_b64, page = future.result()
                    processed_pages += 1
                    progress_bar.progress(processed_pages / total_pages)
                    progress_text.text(f"Processed {processed_pages}/{total_pages} pages")
                    st.session_state.file_results[file_name].append({
                        "page": page_idx,
                        "text": text,
                        "page_time": page_time,
                        "img_b64": img_b64,
                        "image": page
                    })
                except Exception as e:
                    st.error(f"Failed to process {file_name} - Page {page_idx}: {str(e)}")
        
        # Sort results by page number
        for file_name in st.session_state.file_results:
            st.session_state.file_results[file_name].sort(key=lambda x: x["page"])
        
        st.session_state.processed = True
    
    progress_text.text("Processing complete!")

# Sidebar: File selection and summary
if st.session_state.file_results:
    total_pages = sum(st.session_state.debug_info[file]["page_count"] for file in st.session_state.debug_info)
    st.sidebar.write(f"Total files: {len(st.session_state.file_results)}")
    st.sidebar.write(f"Total pages: {total_pages}")
    selected_file = st.sidebar.selectbox("Select a file to preview", options=list(st.session_state.file_results.keys()), index=0)

# Debug info
with st.expander("Debug Information"):
    for file_name, info in st.session_state.debug_info.items():
        st.write(f"PDF: {file_name}")
        st.write(f"Size: {info['pdf_size']} bytes")
        st.write(f"Pages: {info['page_count']}")

# Main content: two columns
if st.session_state.file_results:
    col1, col2 = st.columns([2, 3])
    
    with col1.container():
        st.header("Page Details")
        if selected_file:
            for result in st.session_state.file_results[selected_file]:
                with st.container():
                    st.subheader(f"Page {result['page']}")
                    st.image(result["image"], caption=f"Page {result['page']}", use_container_width=True)
                    st.text_area(f"OCR Result - Page {result['page']}", result["text"], height=150)
                    with st.expander("Page Debug Info"):
                        st.write(f"Processing time: {result['page_time']:.2f} seconds")
                        st.write(f"Base64 string length: {len(result['img_b64'])} characters")
    
    with col2.container():
        st.header("Markdown Preview")
        if selected_file:
            markdown_content = f"# OCR Results for {selected_file}\n\n"
            for result in st.session_state.file_results[selected_file]:
                markdown_content += f"## Page {result['page']}\n\n"
                markdown_content += f"**Processing Time**: {result['page_time']:.2f} seconds\n\n"
                markdown_content += f"```text\n{result['text']}\n```\n\n"
            html_content = markdown2.markdown(markdown_content, extras=["fenced-code-blocks", "tables"])
            st.markdown(f'<div class="markdown-preview">{html_content}</div>', unsafe_allow_html=True)
            
            st.download_button(
                label=f"Download Markdown for {selected_file}",
                data=markdown_content,
                file_name=f"ocr_{selected_file}.md",
                mime="text/markdown"
            )

# Processing summary and combined download
with st.expander("Processing Summary"):
    for file_name, start_time in st.session_state.file_times.items():
        file_time = time.time() - start_time
        st.write(f"Total processing time for {file_name}: {file_time:.2f} seconds")
    
    all_results = []
    for file_name in st.session_state.file_results:
        for result in st.session_state.file_results[file_name]:
            all_results.append(f"File: {file_name} - Page {result['page']}\n{result['text']}")
    if all_results:
        full_text = "\n\n".join(all_results)
        st.download_button(
            label="Download Combined OCR Text",
            data=full_text,
            file_name="combined_ocr_output.txt",
            mime="text/plain"
        )
