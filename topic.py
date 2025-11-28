import time
import json
import requests
import pandas as pd

# ========= CONFIG – EDIT THESE =========
MODEL_NAME = "llama3"  # your model name
BASE_URL = "http://localhost:4000/v1/chat/completions"  # <-- PUT YOUR URL HERE

SUMMARY_EXCEL_PATH = "output_with_summary.xlsx"   # <-- 7 sütunlu excel (summary 7. sütunda)
CATEGORY_EXCEL_PATH = "categories.xlsx"           # <-- 0: category_name, 1: sub_category_name
OUTPUT_EXCEL_PATH = "output_with_topics.xlsx"     # <-- yeni excel

VERIFY_SSL = False
MAX_RETRIES = 2
TIMEOUT_SECONDS = 120
# ======================================

HEADERS = {
    "Content-Type": "application/json"
}

def load_categories():
    """
    Reads category Excel:
    - col 0: main category
    - col 1: sub category (unique)
    Returns:
      subcategories: list[str]
      sub_to_main: dict[sub_category -> main_category]
    """
    df_cat = pd.read_excel(CATEGORY_EXCEL_PATH)

    if df_cat.shape[1] < 2:
        raise ValueError("Category Excel must have at least 2 columns: main category and sub category.")

    main_series = df_cat.iloc[:, 0].astype(str).str.strip()
    sub_series = df_cat.iloc[:, 1].astype(str).str.strip()

    sub_to_main = {}
    for main, sub in zip(main_series, sub_series):
        sub_to_main[sub] = main  # sub categories are unique as you said

    subcategories = list(sub_to_main.keys())
    return subcategories, sub_to_main

def build_classification_messages(summary_text: str, subcategories: list[str]):
    """
    Builds the messages for LLM classification.
    We give the list of available subcategories, and ask for top 3 in JSON.
    """
    subcat_list_text = "\n".join(f"- {s}" for s in subcategories)

    system_msg = (
        "You are an assistant that classifies call summaries into predefined subcategories. "
        "You MUST choose exactly 3 DISTINCT subcategories from the given list. "
        "Return ONLY valid JSON, with no extra text."
    )

    user_msg = f"""
Here is the call summary:

\"\"\"{summary_text}\"\"\"

Here is the list of AVAILABLE subcategories (you MUST select only from these, do not invent new ones):

{subcat_list_text}

Task:
- Select the 3 most relevant subcategories for this summary.
- They must be distinct and come from the list above.
- Return ONLY valid JSON in this exact format:

{{
  "subcategories": [
    "sub_category_name_1",
    "sub_category_name_2",
    "sub_category_name_3"
  ]
}}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def call_llm(messages):
    """
    Generic LLM caller with retries.
    """
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
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            last_error = e
            print(f"[WARN] LLM request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1)

    return f"[LLM_ERROR] {last_error}"

def classify_summary(summary_text: str, subcategories: list[str]) -> list[str]:
    """
    Takes one summary, returns list of up to 3 subcategory names chosen by LLM.
    """
    if not summary_text or summary_text.strip() == "":
        return []

    messages = build_classification_messages(summary_text, subcategories)
    raw_content = call_llm(messages)

    # Try to parse JSON robustly
    try:
        # In case model adds some extra text by mistake, extract JSON substring
        text = raw_content.strip()
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            text = text[first_brace:last_brace + 1]

        parsed = json.loads(text)
        sub_list = parsed.get("subcategories", [])

        # Ensure it's a list of strings
        sub_list = [str(s).strip() for s in sub_list if str(s).strip()]

        # Exactly 3: trim or pad with empty strings
        if len(sub_list) > 3:
            sub_list = sub_list[:3]
        elif len(sub_list) < 3:
            sub_list += [""] * (3 - len(sub_list))

        return sub_list

    except Exception as e:
        print(f"[ERROR] Failed to parse JSON from LLM response: {e}")
        print(f"Raw content was: {raw_content}")
        # Return placeholders to keep row alignment
        return ["", "", ""]

def main():
    # Load categories
    print(f"Reading category Excel: {CATEGORY_EXCEL_PATH}")
    subcategories, sub_to_main = load_categories()
    print(f"Loaded {len(subcategories)} subcategories.")

    # Load summary Excel
    print(f"Reading summary Excel: {SUMMARY_EXCEL_PATH}")
    df = pd.read_excel(SUMMARY_EXCEL_PATH)

    # Need at least 7 columns (summary in 7th)
    if df.shape[1] < 7:
        raise ValueError("Summary Excel must have at least 7 columns (summary at column index 6).")

    summary_series = df.iloc[:, 6]  # 7th column (0-based index 6)

    sub1_list, main1_list = [], []
    sub2_list, main2_list = [], []
    sub3_list, main3_list = [], []

    print("Starting classification...")
    for idx, value in summary_series.items():
        summary_text = "" if pd.isna(value) else str(value)

        chosen_subs = classify_summary(summary_text, subcategories)

        s1, s2, s3 = chosen_subs if len(chosen_subs) == 3 else ["", "", ""]

        m1 = sub_to_main.get(s1, "") if s1 else ""
        m2 = sub_to_main.get(s2, "") if s2 else ""
        m3 = sub_to_main.get(s3, "") if s3 else ""

        sub1_list.append(s1)
        main1_list.append(m1)
        sub2_list.append(s2)
        main2_list.append(m2)
        sub3_list.append(s3)
        main3_list.append(m3)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows...")

    # Build final DataFrame: original 7 cols + 6 new cols
    output_df = df.copy()
    output_df["sub_category_1"] = sub1_list
    output_df["main_category_1"] = main1_list
    output_df["sub_category_2"] = sub2_list
    output_df["main_category_2"] = main2_list
    output_df["sub_category_3"] = sub3_list
    output_df["main_category_3"] = main3_list

    print(f"Writing output Excel: {OUTPUT_EXCEL_PATH}")
    output_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
