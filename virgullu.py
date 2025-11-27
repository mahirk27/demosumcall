import time
import requests
import pandas as pd

# ========= CONFIG – EDIT THESE =========
MODEL_NAME = "llama3"  # your model name
BASE_URL = "http://localhost:4000/v1/chat/completions"  # <-- PUT YOUR URL HERE

INPUT_EXCEL_PATH = "input_single_column.xlsx"          # <-- ORIGINAL EXCEL (ONE COLUMN)
OUTPUT_EXCEL_PATH = "output_structured_with_summary.xlsx"  # <-- NEW EXCEL WITH 7 COLUMNS

VERIFY_SSL = False      # you said verify_ssl = False
MAX_RETRIES = 2
TIMEOUT_SECONDS = 120
# ======================================

HEADERS = {
    "Content-Type": "application/json"
}

def build_messages(transcript: str):
    """
    You can adjust the prompt here.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant that summarizes phone call transcripts. "
                "Write a clear, concise summary of the call in 3–5 sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Here is the call transcript:\n\n{transcript}\n\nPlease provide only the summary.",
        },
    ]

def summarize_with_llm(transcript: str) -> str:
    """
    Sends the transcript to the LLM and returns the summary text.
    Uses MAX_RETRIES and TIMEOUT_SECONDS.
    """
    messages = build_messages(transcript)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                BASE_URL,
                json=payload,
                headers=HEADERS,
                verify=VERIFY_SSL,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

            # OpenAI-style response format (LiteLLM usually follows this)
            summary = data["choices"][0]["message"]["content"]
            return summary.strip()

        except Exception as e:
            last_error = e
            print(f"[WARN] LLM request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)  # small delay before retry

    # If all retries failed, return an error marker (you can change this)
    return f"[SUMMARY_ERROR] {last_error}"

def main():
    # Read the original Excel (one column)
    print(f"Reading input Excel: {INPUT_EXCEL_PATH}")
    df = pd.read_excel(INPUT_EXCEL_PATH)

    if df.shape[1] < 1:
        raise ValueError("Input Excel must have at least 1 column.")

    # Use the first (and only) column
    single_col_series = df.iloc[:, 0]

    first_six_columns_rows = []  # list of [c1, c2, c3, c4, c5, c6]
    summaries = []

    print("Starting parsing and summarization...")
    for idx, value in single_col_series.items():
        row_text = "" if pd.isna(value) else str(value)

        # Split by comma
        parts = [p.strip() for p in row_text.split(",")]

        # Ensure we have at least 6 elements (pad with empty strings if shorter)
        if len(parts) < 6:
            parts += [""] * (6 - len(parts))

        first_six = parts[:6]
        transcript = first_six[5]  # 6th element = transcript

        if not transcript:
            summary = ""  # or "No transcript"
        else:
            summary = summarize_with_llm(transcript)

        first_six_columns_rows.append(first_six)
        summaries.append(summary)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows...")

    # Build structured DataFrame:
    # 6 columns extracted + 1 summary column
    output_df = pd.DataFrame(
        first_six_columns_rows,
        columns=["col1", "col2", "col3", "col4", "col5", "col6"],
    )
    output_df["summary"] = summaries

    # Save to new Excel
    print(f"Writing output Excel: {OUTPUT_EXCEL_PATH}")
    output_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
