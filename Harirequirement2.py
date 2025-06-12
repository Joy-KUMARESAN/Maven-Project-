import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import csv
import os
import re
import requests
import numpy as np
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    st.error("sentence-transformers not installed. Please run: pip install sentence-transformers")
    st.stop()

try:
    from langchain_anthropic import Anthropic
    setattr(Anthropic, "count_tokens", lambda self, text: len(text.split()))
    from langchain_anthropic import ChatAnthropic
except ImportError:
    st.error("langchain-anthropic not installed. Please run: pip install langchain-anthropic")
    st.stop()

try:
    from browser_use import Agent
except ImportError:
    st.error("browser_use not installed or not found.")
    st.stop()

load_dotenv()

UNWANTED_PATTERNS = [
    r'error\s*=\s*None',
    r'include_in_memory\s*=\s*False',
    r'all_model_outputs\s*=\s*\[.*',
    r'DOMHistoryElement\(',
    r'\],\s*all_model_outputs',
    r'\{.*\}',
    r'\[.*\]',
    r'Playwright not supported',
    r'Page not found',
    r'404 error',
    r'^\s*$',
    r"success['\"]?\s*:\s*True",
    r"interacted_element['\"]?\s*:\s*None",
    r"done['\"]?\s*:\s*\{.*",
    r"agent\s*:",
    r"extract_content\s*:",
    r"\),",
]

def clean_field(text):
    if not text:
        return text
    for pat in UNWANTED_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_hs_codes(csv_file):
    hs_data = []
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hs_data.append((row['HS Code'], row['Product Description']))
        return hs_data
    except Exception as e:
        st.error(f"Failed to load HS codes: {e}")
        return []

def correct_address(address):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return "API key missing"
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    try:
        response = requests.get(url, timeout=10)
        results = response.json()
        if results.get('status') == 'OK' and results.get('results'):
            return results['results'][0]['formatted_address']
        else:
            return "No valid address found"
    except Exception as e:
        return f"Address correction error: {e}"

def match_hs_code(user_desc, hs_data, model):
    try:
        descriptions = [desc for _, desc in hs_data]
        corpus_embeddings = model.encode(descriptions, convert_to_tensor=True)
        query_embedding = model.encode(user_desc, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_result = int(np.argmax(cos_scores))
        matched_hs_code, matched_description = hs_data[top_result]
        similarity_score = float(cos_scores[top_result])
        return matched_hs_code, similarity_score, matched_description
    except Exception as e:
        return "", 0.0, f"HS code matching error: {e}"

async def get_product_info(product_name: str):
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022", timeout=4000, stop=["bye"])
    task_description = f"""
You are a browsing agent. Your task is to go to the website "https://www.all-ett.com" and extract information about the product named "{product_name}".
Please follow these exact steps:
1. Open the homepage of All-Ett: https://www.all-ett.com
2. Wait until the page loads. If there's a popup or newsletter, dismiss it by clicking the X or cancel button.
3. Locate and click on the search icon (usually a magnifying glass in the top right).
4. In the search bar that appears, type: {product_name}, then press Enter.
5. From the search results, click on the FIRST product image or link that appears.
6. On the product page, extract the following:
   - **Title**: The full product name (look for <h1> or .product-title.h3)
   - **Product description (extract all text content located at the XPath: /html/body/main/section[1]/div/div/product-rerender/div/safe-sticky/div/div[4]/div/p)
   - **Price**: Identify the exact product price shown on the page.
7. Return the extracted data clearly formatted like this:
Be concise. Do not include unrelated page text.
output format:
Title: <Product Title>
Description: <Product Description>
Price: <Product Price>
    """
    agent = Agent(task=task_description, llm=llm, use_vision=True)
    try:
        result = await agent.run()
        result_text = str(result)
        title_match = re.search(r'"product_title"\s*:\s*"([^"]+)"', result_text)
        if title_match:
            product_title = title_match.group(1).strip()
        else:
            title_match = re.search(r'Title:\s*(.*?)(?:\nDescription:|$)', result_text, re.DOTALL)
            if title_match:
                product_title = title_match.group(1).strip()
            else:
                title_match = re.search(r'^(.*?)(?:\nDescription:|$)', result_text, re.DOTALL)
                product_title = title_match.group(1).strip() if title_match else None
        product_title = clean_field(product_title)
        desc_match = re.search(r'"description":\s*"([^"]+)"', result_text)
        if desc_match:
            product_description = desc_match.group(1).strip()
        else:
            desc_match = re.search(r'Description:\s*([\s\S]*?)(?:\nPrice:|$)', result_text)
            product_description = desc_match.group(1).strip() if desc_match else None
        product_description = clean_field(product_description)
        price_match = re.search(r'"price"\s*:\s*"([^"]+)"', result_text)
        if price_match:
            price = price_match.group(1).strip()
        else:
            price_match = re.search(r'Price:\s*([^\n",]+)', result_text)
            price = price_match.group(1).strip() if price_match else None
        if price:
            price = re.split(r'["\',]', price)[0].strip()
        price = clean_field(price)
        data = {
            "title": product_title,
            "description": product_description,
            "price": price,
        }
        return data
    except Exception as e:
        return {"error": f"Product info error: {e}"}
    finally:
        try:
            await agent.close()
        except Exception:
            pass

def ollama_chat(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[Ollama error: {e}]"

def process_uploaded_file(df, hs_data, model):
    results = []
    for idx, row in df.iterrows():
        product_name = row.get('product name', '').strip()
        address = row.get('address', '').strip()
        try:
            product_info = asyncio.run(get_product_info(product_name))
            if "error" in product_info:
                raise Exception(product_info["error"])
            matched_hs_code, similarity, matched_desc = match_hs_code(
                product_info.get("description", ""), hs_data, model
            )
            corrected_address = correct_address(address) if address else ""
            result = {
                "Product Name": product_name,
                "Product Title": product_info.get("title", ""),
                "Price": product_info.get("price", ""),
                "Original Address": address,
                "Corrected Address": corrected_address,
                "Product Description": product_info.get("description", ""),
                "Matched HS Code": matched_hs_code,
                "Similarity Score": f"{similarity:.4f}",
                "Matched Description": matched_desc,
                "Error": ""
            }
            results.append(result)
        except Exception as e:
            results.append({
                "Product Name": product_name,
                "Product Title": "",
                "Price": "",
                "Original Address": address,
                "Corrected Address": "",
                "Product Description": "",
                "Matched HS Code": "",
                "Similarity Score": "",
                "Matched Description": "",
                "Error": str(e)
            })
    return pd.DataFrame(results)

def parse_user_intent(user_input, product_names):
    text = user_input.lower()
    matched_product = None
    for name in sorted(product_names, key=len, reverse=True):
        if name.lower() in text:
            matched_product = name
            break
    if "result" in text or "download" in text or "csv" in text:
        return "download", None
    if "description" in text and matched_product:
        return "description", matched_product
    if ("title" in text or "name" in text) and matched_product:
        return "title", matched_product
    if "address" in text and matched_product:
        return "address", matched_product
    # Fallback for non-table products
    if "description" in text and "for" in text:
        prod_name = user_input.lower().split("for",1)[1].strip()
        return "description", prod_name
    if "address" in text and "for" in text:
        prod_name = user_input.lower().split("for",1)[1].strip()
        return "address", prod_name
    return None, None

def chat_bubble(sender, message):
    if sender == "bot":
        st.markdown(f'<div class="chat-bubble-bot">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-user">{message}</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Description and Address Agent", layout="wide")
    st.markdown("""
    <style>
    body, .stApp { background-color: #f6f7fa; color: #222; }
    .stTextInput>div>div>input { background-color: #f6f7fa; color: #222; }
    .stDataFrame { background-color: #f6f7fa; }
    .chat-bubble-bot {
        background: #e9ecef;
        color: #222;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        margin-right: 35%;
        border-left: 6px solid #2b7cff;
    }
    .chat-bubble-user {
        background: #2b7cff;
        color: #fff;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        margin-left: 35%;
        text-align: right;
    }
    .example-box {
        background: #f1f5fb;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.96em;
        border-left: 4px solid #2b7cff;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Description and Address Agent")

    hs_code_csv = os.getenv("HS_CODE_CSV")
    if not hs_code_csv or not os.path.isfile(hs_code_csv):
        st.error("HS_CODE_CSV path not found or file does not exist. Please set the HS_CODE_CSV environment variable.")
        return

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

    hs_data = load_hs_codes(hs_code_csv)
    if not hs_data:
        st.error("Failed to load HS code data.")
        return

    if "conv_stage" not in st.session_state:
        st.session_state["conv_stage"] = "greet"
        st.session_state["chat_history"] = []
        st.session_state["results_df"] = pd.DataFrame(columns=[
            "Product Name", "Product Title", "Price", "Original Address", "Corrected Address",
            "Product Description", "Matched HS Code", "Similarity Score", "Matched Description", "Error"
        ])
        st.session_state["download_ready"] = False
        st.session_state["last_processed_input"] = ""
        st.session_state["pending_update"] = None
        st.session_state["csv_processed"] = False

    left, right = st.columns([1,2])

    with left:
        st.header("Actions")
        uploaded_file = st.file_uploader("Upload CSV (columns: 'product name', 'address')", type=["csv"])
        st.markdown("<div class='example-box'><b>Example commands:</b><br>"
                    "- Give me the product description for Blue Wallet<br>"
                    "- Change the description for Black Wallet<br>"
                    "- Correct the address for 123 Main St<br>"
                    "- Download the results</div>", unsafe_allow_html=True)
        with st.form("side_chat_form", clear_on_submit=True):
            user_input = st.text_input("Type a command...", key="side_chat")
            submitted = st.form_submit_button("Send")
        if st.session_state["csv_processed"]:
            if st.button("Reset for new upload"):
                st.session_state["csv_processed"] = False
                st.session_state["results_df"] = pd.DataFrame(columns=[
                    "Product Name", "Product Title", "Price", "Original Address", "Corrected Address",
                    "Product Description", "Matched HS Code", "Similarity Score", "Matched Description", "Error"
                ])
                st.session_state["conv_stage"] = "greet"
                st.rerun()

    with right:
        st.header("Results and Chat")
        if st.session_state["results_df"] is not None and not st.session_state["results_df"].empty:
            st.dataframe(st.session_state["results_df"], use_container_width=True)
        for sender, message in st.session_state["chat_history"]:
            chat_bubble(sender, message)

    # --- Handle Upload ---
    if uploaded_file and not st.session_state["csv_processed"]:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return
        st.success("✅ CSV uploaded successfully!")
        with st.spinner("Processing uploaded data..."):
            st.session_state["results_df"] = process_uploaded_file(df, hs_data, model)
        st.session_state["conv_stage"] = "results"
        st.session_state["download_ready"] = False
        st.session_state["csv_processed"] = True
        st.session_state["chat_history"].append(("bot", "Products processed. You may now ask for improvements or request your results."))
        st.rerun()

    # --- Handle Chat ---
    if submitted and user_input and user_input.strip():
        st.session_state["chat_history"].append(("user", user_input))
        results_df = st.session_state["results_df"]
        product_names = list(results_df["Product Name"].dropna().unique())
        intent, prod_name = parse_user_intent(user_input, product_names)
        if intent == "description" and prod_name:
            if prod_name in results_df["Product Name"].values:
                # Rephrase using local Mistral, only 5 important words
                orig_desc = results_df.loc[results_df["Product Name"] == prod_name, "Product Description"].iloc[0]
                prompt = (
                    f"Take the following product description and return only the 5 most important words, "
                    f"separated by spaces, no punctuation or extra text:\n\n{orig_desc}"
                )
                new_desc = ollama_chat(prompt, model="mistral")
                msg = (
                    f"Description for <b>{prod_name}</b> proposed.<br><br>"
                    f"<b>Original:</b><br>{orig_desc}<br><br>"
                    f"<b>5 Important Words:</b><br>{new_desc}"
                )
                st.session_state["pending_update"] = {
                    "prod_name": prod_name,
                    "field": "Product Description",
                    "new_value": new_desc,
                    "msg": msg,
                }
                st.rerun()
            else:
                # Not in table: scrape and show ONLY in chat, not in table
                with st.spinner(f"Scraping product info for {prod_name}..."):
                    info = asyncio.run(get_product_info(prod_name))
                if "error" in info:
                    st.session_state["chat_history"].append(("bot", f"❌ {info['error']}"))
                else:
                    st.session_state["chat_history"].append((
                        "bot",
                        f"<b>Product Title:</b> {info.get('title','')}<br>"
                        f"<b>Description:</b> {info.get('description','')}<br>"
                        f"<b>Price:</b> {info.get('price','')}"
                    ))
                st.rerun()
        elif intent == "address" and prod_name:
            if prod_name not in results_df["Original Address"].values:
                with st.spinner("Correcting address..."):
                    corrected = correct_address(prod_name)
                st.session_state["chat_history"].append(("bot", f"<b>Corrected Address:</b> {corrected}"))
                st.rerun()
            else:
                idx = results_df[results_df["Original Address"] == prod_name].index[0]
                orig_addr = results_df.at[idx, "Original Address"]
                new_addr = correct_address(orig_addr)
                msg = (
                    f"Address for <b>{prod_name}</b> proposed.<br><br>"
                    f"<b>Original:</b><br>{orig_addr}<br><br>"
                    f"<b>Corrected:</b><br>{new_addr}"
                )
                st.session_state["pending_update"] = {
                    "prod_name": prod_name,
                    "field": "Corrected Address",
                    "new_value": new_addr,
                    "msg": msg,
                }
                st.rerun()
        elif intent == "download":
            st.session_state["download_ready"] = True
            st.session_state["chat_history"].append(("bot", "Here are your results. Click below to download your CSV."))
            st.rerun()
        else:
            st.session_state["chat_history"].append(("bot", "Sorry, I couldn't understand your request."))
            st.rerun()

    # --- Accept/Reject/Download logic ---
    results_df = st.session_state["results_df"]
    pending = st.session_state.get("pending_update", None)
    if pending and not results_df.empty:
        with right:
            st.markdown(f"**Proposed Change:**<br>{pending['msg']}", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            accept = col1.button("Accept", key="accept_btn")
            reject = col2.button("Reject", key="reject_btn")
            if accept:
                idx = results_df[results_df["Product Name"] == pending["prod_name"]].index[0]
                results_df.at[idx, pending["field"]] = pending["new_value"]
                st.session_state["results_df"] = results_df
                st.session_state["chat_history"].append(("bot", f"✅ Change accepted for <b>{pending['prod_name']}</b>."))
                st.session_state["pending_update"] = None
                st.rerun()
            elif reject:
                st.session_state["chat_history"].append(("bot", f"❌ Change rejected for <b>{pending['prod_name']}</b>."))
                st.session_state["pending_update"] = None
                st.rerun()

    if st.session_state.get("download_ready", False) and results_df is not None and not results_df.empty:
        with right:
            st.download_button(
                label="💾 Download Results",
                data=st.session_state["results_df"].to_csv(index=False).encode('utf-8'),
                file_name="processed_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
