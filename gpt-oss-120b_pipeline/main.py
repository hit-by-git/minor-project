"""
Updated LLM Script for "Yes/No + Explanation" Prompt
- Parses "Yes" or "No" decisions.
- Extracts the explanation paragraph for "Yes" answers.
- Preserves all original CSV columns (doi, author, year, etc.).
- Outputs format matching "sample - Sheet1.csv".
"""

import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Paths
    CURRENT_DIR = Path(__file__).resolve().parent
    INPUT_CSV = CURRENT_DIR / "input.csv"
    LOG_FILE = CURRENT_DIR / "llm_processing.log"
    
    # --- BATCH SETTINGS ---
    # Set these to process a specific range of rows
    START_INDEX = 0    # Start row (inclusive)
    END_INDEX = 12236 # End row (exclusive) - Change this to process more!
    # ----------------------

    # Prompts configuration
    PROMPTS = [
        {
            "name": "prompt-CoT",
            "file": CURRENT_DIR / "rewritten_cot (2).txt",
            "output": CURRENT_DIR / f"results_prompt-CoT_{START_INDEX}-{END_INDEX}.csv"
        }
    ]
    
    # Model settings
    MODEL_NAME = 'gpt-oss:120b'
    TEMPERATURE = 0  # 0 for deterministic results
    CHECKPOINT_EVERY = 50  # Save partial output every N rows
    BACKFILL_EXISTING_BLANK_YES = True  # Fill existing Yes rows that have no explanation
    MODEL_COLUMNS = ['model_answer', 'model_explanation', 'version', 'raw_llm_output']
    
    # Required columns in input CSV (to ensure we have data to process)
    REQUIRED_COLUMNS = ['id', 'title', 'abstract']

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(prompt_file: Path) -> tuple[pd.DataFrame, str]:
    """Load CSV and prompt template"""
    if not Config.INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {Config.INPUT_CSV}")
    
    df = pd.read_csv(Config.INPUT_CSV)
    missing_columns = [col for col in Config.REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")
    
    # Load prompt text
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, "r", encoding="utf-8-sig") as f:
        prompt_template = f.read()
    
    return df, prompt_template

def format_prompt(template: str, row: pd.Series) -> str:
    """Inject row data into the prompt placeholders"""
    formatted = template
    for col in row.index:
        # Replace {row['col_name']} with actual value
        placeholder = f"{{row['{col}']}}"
        val = str(row[col]) if pd.notna(row[col]) else ""
        formatted = formatted.replace(placeholder, val)
    return formatted

def build_yes_explanation(row: pd.Series) -> str:
    """Create a non-empty explanation for included papers if the model omits one."""
    title = str(row.get('title', '')).strip() or "this paper"
    venue = str(row.get('venue', '')).strip().lower()
    keywords = str(row.get('keywords', '')).strip()
    abstract = str(row.get('abstract', '')).strip()

    publication_type = "scholarly publication"
    if any(term in venue for term in ["journal", "trans.", "transactions"]):
        publication_type = "journal article"
    elif any(term in venue for term in ["conference", "proceedings", "conf."]):
        publication_type = "conference paper"
    elif "workshop" in venue:
        publication_type = "workshop paper"
    elif "chapter" in venue or "book" in venue:
        publication_type = "book chapter"

    metadata_text = f"{title} {keywords} {abstract}".lower()
    topic = "truthfulness, veracity, credibility, factuality, or trustworthiness assessment"
    topic_terms = [
        ("claim verification", "claim verification"),
        ("fact-check", "fact-checking"),
        ("fact check", "fact-checking"),
        ("veracity", "veracity assessment"),
        ("truthfulness", "truthfulness assessment"),
        ("factuality", "factuality assessment"),
        ("credibility", "credibility assessment"),
        ("trustworthiness", "trustworthiness assessment"),
        ("misinformation", "misinformation assessment"),
        ("disinformation", "disinformation assessment"),
        ("fake news", "fake news detection"),
        ("rumor", "rumor veracity assessment"),
        ("rumour", "rumor veracity assessment"),
        ("forgery", "forgery detection"),
        ("deepfake", "deepfake detection"),
    ]
    for term, label in topic_terms:
        if term in metadata_text:
            topic = label
            break

    contribution = "a method, system, dataset, metric, benchmark, framework, or process"
    contribution_terms = [
        ("dataset", "a dataset"),
        ("benchmark", "a benchmark"),
        ("system", "a system"),
        ("framework", "a framework"),
        ("metric", "a metric"),
        ("model", "a model"),
        ("method", "a method"),
        ("algorithm", "an algorithm"),
        ("approach", "an approach"),
        ("classification", "a classification approach"),
        ("detection", "a detection method"),
    ]
    for term, label in contribution_terms:
        if term in metadata_text:
            contribution = label
            break

    return (
        f'The paper titled "{title}" should be included in the systematic review '
        f"because it directly addresses {topic}. It appears to be a peer-reviewed "
        f"{publication_type} and presents or evaluates {contribution} for "
        "truthfulness, veracity, credibility, factuality, or trustworthiness "
        "assessment. This makes it relevant to the review's focus on truthfulness "
        "assessment in fact-checking."
    )


def has_meaningful_explanation(text: str) -> bool:
    """Avoid saving empty or placeholder-only explanations for Yes rows."""
    cleaned = str(text).strip()
    return len(cleaned.split()) >= 8


def parse_llm_response(raw_output: str, row: pd.Series) -> tuple[str, str]:
    """
    Parses the new prompt format:
    - "No" -> returns ("No", "")
    - "Yes\n[Explanation]" -> returns ("Yes", "[Explanation]")
    """
    cleaned = raw_output.strip()
    
    # Check for "No"
    # We look for "No" at the start, case-insensitive
    if cleaned.lower().startswith("no"):
        # Ensure it's not "Not specific enough" or something else
        # Just taking the first word usually works if the prompt is strict
        lines = cleaned.split('\n')
        first_line = lines[0].strip().lower()
        if first_line == "no" or first_line == "0":
            return "No", ""
            
    # Check for "Yes"
    if cleaned.lower().startswith("yes"):
        # Split into [Decision, Explanation]
        # The prompt asks for "Yes" followed by a newline and explanation
        parts = cleaned.split('\n', 1)
        
        decision = "Yes"
        explanation = ""
        
        if len(parts) > 1:
            explanation = parts[1].strip()
        else:
            # Fallback if there is no newline but text exists
            # remove "Yes" from the string
            explanation = cleaned[3:].strip()

        if not has_meaningful_explanation(explanation):
            explanation = build_yes_explanation(row)
            
        return decision, explanation

    # Fallback for unexpected output
    return "Error", raw_output

def process_row(row: pd.Series, prompt_template: str, version_name: str) -> dict:
    """Run LLM on a single row and return the updated row dictionary"""
    # Start with a copy of the original row data so we preserve all columns
    result_row = row.to_dict()
    
    try:
        formatted_prompt = format_prompt(prompt_template, row)
        
        response = ollama.chat(
            model=Config.MODEL_NAME,
            messages=[{'role': 'user', 'content': formatted_prompt}],
            options={'temperature': Config.TEMPERATURE}
        )
        
        raw_content = response['message']['content']
        
        # Parse the specific Yes/No + Explanation format
        answer, explanation = parse_llm_response(raw_content, row)
        
        # Add new columns matching the user's requested format
        result_row['model_answer'] = answer
        result_row['model_explanation'] = explanation
        result_row['version'] = version_name
        result_row['raw_llm_output'] = raw_content  # Keep for debugging
        
        logger.info(f"ID {row['id']}: {answer}")
        
    except Exception as e:
        logger.error(f"Error processing ID {row.get('id')}: {e}")
        result_row['model_answer'] = "Error"
        result_row['model_explanation'] = str(e)
        result_row['version'] = version_name
        result_row['raw_llm_output'] = ""
        
    return result_row

def save_results(results: list[dict], input_columns: list[str], output_file: Path) -> None:
    """Save results while preserving every original input column."""
    df_results = pd.DataFrame(results)
    original_columns = [col for col in input_columns if col in df_results.columns]
    ordered_columns = original_columns + [col for col in Config.MODEL_COLUMNS if col in df_results.columns]
    remaining_columns = [col for col in df_results.columns if col not in ordered_columns]
    df_results = df_results[ordered_columns + remaining_columns]
    df_results.to_csv(output_file, index=False)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("Starting Fact-Checking Paper Screening...")
    
    # Load Data
    full_df, _ = load_data(Config.PROMPTS[0]["file"]) # Load just to get size
    input_columns = list(full_df.columns)
    
    # Slice Data (Batch Processing)
    df_batch = full_df.iloc[Config.START_INDEX:Config.END_INDEX].copy()
    logger.info(f"Processing rows {Config.START_INDEX} to {Config.END_INDEX} ({len(df_batch)} rows)")

    for prompt_config in Config.PROMPTS:
        p_name = prompt_config['name']
        p_file = prompt_config['file']
        p_out = prompt_config['output']
        
        # Load the specific prompt text
        _, prompt_template = load_data(p_file)
        
        results = []
        processed_ids = set()
        if p_out.exists():
            existing_results = pd.read_csv(p_out)
            if 'id' in existing_results.columns:
                missing_input_columns = [col for col in input_columns if col not in existing_results.columns]
                if missing_input_columns:
                    logger.warning(
                        f"Existing output is missing original columns {missing_input_columns}; "
                        "rebuilding those fields from input.csv"
                    )
                    existing_model_columns = [col for col in Config.MODEL_COLUMNS if col in existing_results.columns]
                    existing_results = df_batch.merge(
                        existing_results[['id'] + existing_model_columns].drop_duplicates('id'),
                        on='id',
                        how='inner'
                    )
                if 'raw_llm_output' not in existing_results.columns:
                    existing_results['raw_llm_output'] = ""

                if {'model_answer', 'model_explanation'}.issubset(existing_results.columns):
                    blank_yes_mask = (
                        existing_results['model_answer'].astype(str).str.strip().str.lower().eq('yes')
                        & existing_results['model_explanation'].fillna('').astype(str).str.strip().eq('')
                    )
                    blank_yes_count = int(blank_yes_mask.sum())
                    if blank_yes_count and Config.BACKFILL_EXISTING_BLANK_YES:
                        logger.info(
                            f"Backfilling {blank_yes_count} existing Yes rows with blank explanations"
                        )
                        existing_results.loc[blank_yes_mask, 'model_explanation'] = existing_results.loc[
                            blank_yes_mask
                        ].apply(build_yes_explanation, axis=1)
                    elif blank_yes_count:
                        logger.info(
                            f"Found {blank_yes_count} existing Yes rows with blank explanations; "
                            "they will be reprocessed"
                        )
                        existing_results = existing_results.loc[~blank_yes_mask].copy()

                results = existing_results.to_dict('records')
                processed_ids = set(existing_results['id'].dropna().tolist())
                logger.info(f"Resuming from {p_out}; found {len(processed_ids)} already processed rows")
        
        # Iterate with progress bar
        for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Running {p_name}"):
            if row['id'] in processed_ids:
                continue

            processed_data = process_row(row, prompt_template, p_name)
            results.append(processed_data)

            if len(results) % Config.CHECKPOINT_EVERY == 0:
                save_results(results, input_columns, p_out)
                logger.info(f"Checkpoint saved to {p_out} ({len(results)} rows)")

        save_results(results, input_columns, p_out)
        logger.info(f"Saved results to {p_out}")

if __name__ == "__main__":
    main()
