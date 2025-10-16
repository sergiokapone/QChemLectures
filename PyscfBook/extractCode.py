import re
import os


def process_latex_file(input_file, output_file, code_dir="code"):
    """
    Processes a LaTeX file to extract minted python blocks, save them as .py files,
    and replace the blocks with \inputcode{filename.py}.

    :param input_file: Path to input LaTeX file
    :param output_file: Path to output LaTeX file
    :param code_dir: Directory to save .py files (created if not exists)
    """
    # Create code directory if it doesn't exist
    os.makedirs(code_dir, exist_ok=True)

    counter = 1  # Moved before the nested function definition

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to match \begin{minted}{python} ... \end{minted}
    pattern = r"\\begin\{minted\}\{python\}\s*(.*?)\\end\{minted\}"

    def replace_minted(match):
        nonlocal counter
        code = match.group(1).strip()
        # Clean up code: remove extra newlines, etc.
        code_lines = [line.strip() for line in code.split("\n") if line.strip()]
        code = "\n".join(code_lines)

        # Generate filename
        filename = f"{code_dir}/code_{counter}.py"
        counter += 1

        # Save code to file
        with open(filename, "w", encoding="utf-8") as py_file:
            py_file.write(code)

        return f"\\inputcode{{{os.path.basename(filename)}}}"

    # Replace all matches
    modified_content = re.sub(pattern, replace_minted, content, flags=re.DOTALL)

    # Write modified content
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Processed {counter - 1} code blocks. Output: {output_file}")


if __name__ == "__main__":
    # Usage example:
    process_latex_file("WelcomeWithPyscf.tex", "output.tex")
