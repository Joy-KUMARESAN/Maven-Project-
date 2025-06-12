# Description and Address Agent

This Streamlit application automates the extraction of product information from [all-ett.com], corrects addresses using the Google Maps API, matches product descriptions to HS codes using semantic search, and provides an interactive chat interface for querying and downloading results.

---

## Features

- **Product Extraction:** Uses an LLM-powered agent to browse and extract product title, description, and price from all-ett.com.
- **Address Correction:** Validates and corrects addresses via the Google Maps Geocoding API.
- **HS Code Matching:** Finds the closest HS code for a product description using sentence embeddings.
- **Interactive Chat:** Allows users to ask for product descriptions, address corrections, and download results.
- **CSV Upload:** Process multiple products at once by uploading a CSV file.

---

## Installation

1. **Clone the repository:**

git clone https://github.com/yourusername/Maven-Project.git
cd Maven-Project


2. **Install dependencies:**
pip install -r requirements.txt


If you see errors about missing packages (such as `sentence-transformers`, `langchain-anthropic`, or `browser_use`), install them individually as needed.

---

## Environment Setup

1. **Create a `.env` file** in the project root (you can copy from `.env.example`):

GOOGLE_MAPS_API_KEY=your_google_maps_api_key
HS_CODE_CSV=path/to/your/hs_codes.csv


- Replace `your_google_maps_api_key` with your actual Google Maps API key.
- Set `HS_CODE_CSV` to the path of your HS codes CSV file.

2. **Prepare your HS codes CSV**

The CSV should have at least these columns:

HS Code,Product Description


---

## Usage

1. **Run the app:**

2. **Using the app:**
- Upload a CSV file with columns: `product name`, `address`
- Use the chat interface to:
  - Get product descriptions
  - Correct addresses
  - Download results

---

## Example Commands

- “Give me the product description for Blue Wallet”
- “Correct the address for 123 Main St”
- “Download the results”

---

## File Structure

.
├── main.py # Streamlit UI and app logic
├── agents.py # LLM agent and browser automation 
├── utils.py # Utility functions (cleaning, matching, etc.)
├── requirements.txt # Python dependencies
├── .env.example # Example environment variables
├── README.md # This file
└── data/
└── hs_codes.csv # (Your HS code data)


---

## Dependencies

- streamlit
- pandas
- numpy
- requests
- python-dotenv
- sentence-transformers
- langchain-anthropic
- browser_use

---

## Notes

- For Google Maps API, see [Get an API Key](https://developers.google.com/maps/documentation/geocoding/get-api-key)
- For best results, use a recent version of Python (3.8+ recommended)
- If you encounter missing module errors, install the required package as instructed in the app

---

