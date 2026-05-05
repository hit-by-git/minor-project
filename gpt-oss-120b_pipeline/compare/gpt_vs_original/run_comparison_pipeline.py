import subprocess
import sys


SCRIPTS = [
    "build_alignment_summary.py",
    "doi_match_yes_rows.py",
    "title_match_fallback.py",
    "build_combined_matches.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n== {script} ==")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
