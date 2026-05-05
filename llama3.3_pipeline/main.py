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
import re
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Paths
    CURRENT_DIR = Path(".")
    INPUT_CSV = CURRENT_DIR / "input.csv"
    LOG_FILE = CURRENT_DIR / "llm_processing.log"
    
    # --- BATCH SETTINGS ---
    # Set these to process a specific range of rows
    START_INDEX = 0    # Start row (inclusive)
    END_INDEX = 6000 # End row (exclusive) - Change this to process more!
    # ----------------------

    # Prompts configuration
    PROMPTS = [
        {
            "name": "prompt-CoT",
            "file": CURRENT_DIR / "prompt-cot.txt",
            "output": CURRENT_DIR / f"results_prompt-CoT_{START_INDEX}-{END_INDEX}.csv"
        }
    ]
    
    # Model settings
    MODEL_NAME = 'llama3.3'
    TEMPERATURE = 0  # 0 for deterministic results
    
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
    
    # Load prompt text
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, "r", encoding="utf-8") as f:
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

def parse_llm_response(raw_output: str) -> tuple[str, str]:
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
        answer, explanation = parse_llm_response(raw_content)
        
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
        
    return result_row

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("Starting Fact-Checking Paper Screening...")
    
    # Load Data
    full_df, _ = load_data(Config.PROMPTS[0]["file"]) # Load just to get size
    
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
        
        # Iterate with progress bar
        for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Running {p_name}"):
            processed_data = process_row(row, prompt_template, p_name)
            results.append(processed_data)
            
        # Create DataFrame from results
        # This automatically aligns columns; keys present in 'results' become columns
        df_results = pd.DataFrame(results)
        
        # Organize columns to match the input CSV column order exactly
        cols = list(df_results.columns)
        priority_cols = ['id', 'doi', 'title', 'author', 'year', 'keywords', 'venue', 'abstract',
                         'sample_prescreening', 'model_answer', 'model_explanation', 'version']
        sorted_cols = [c for c in priority_cols if c in cols]
        df_results = df_results[sorted_cols]
        
        # Save
        df_results.to_csv(p_out, index=False)
        logger.info(f"Saved results to {p_out}")

if __name__ == "__main__":
    main()