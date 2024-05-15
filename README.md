# Codebase Query and Modification Tool

## Overview

This tool leverages OpenAI's GPT-4-turbo and vector databases to allow users to query and modify large Python codebases effectively. The solution is designed to handle codebases that are too large to fit in the context window of modern language models, providing functionalities for querying, summarizing, analyzing, and modifying the code.

## Features

- **Vector Database (VDB) Building**: Populates a VDB with embeddings of code snippets to allow efficient searching.
- **Summarization**: Generates summaries of code files and their components.
- **Abstract Syntax Tree (AST) Parsing**: Analyzes code structure to extract detailed information about classes and functions.
- **Query Functionality**: Allows users to query the codebase and find relevant code snippets.
- **Code Modification**: Suggests and applies code changes based on user input.
- **Command-Line Interface (CLI)**: Provides an easy-to-use CLI for interacting with the tool.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Set up the environment:**
    - Ensure you have Python 3.10+ installed.
    - Install the required Python packages:
        ```sh
        pip install openai faiss-cpu tqdm numpy python-dotenv
        ```

3. **Set up OpenAI API key:**
    - Create a `.env` file in the project directory with your OpenAI API key:
        ```
        OPENAI_API_KEY=your_openai_api_key
        ```

4. **Organize your code folder:**
    - Place your Python code files in a directory named `codefolder` within the project directory.

## Usage

1. **Build the Vector Database (VDB) and Summarize the Codebase:**
    ```sh
    python otherway_vguppgift.py
    ```
    - The script will process the code files, build the VDB, and generate summaries.

2. **Interact with the CLI:**
    - Available commands:
        1. `query <question>`: Query the codebase for relevant code snippets.
        2. `summary <file_path>`: Get the summary of a specified file.
        3. `ast <file_path>`: Get the AST representation of a specified file.
        4. `suggest <file_path> <prompt>`: Get code modification suggestions based on a prompt.
        5. `apply <file_path> <changes>`: Apply suggested changes to the specified file.
        6. `exit`: Exit the CLI.

    - Example usage:
        ```sh
        Enter a command: query where is factorial function
        ```

## Example Outputs

- **Query Command:**
    ```
    Enter a command: query where is factorial function
    Relevant files and code snippets:
    File: ./codefolder/mat_code.py
    Function: factorial
    Docstring: This function calculates the factorial of a number.
    Lines: 10-15
    Distance: 0.445422887802124
    ----------------------------------------
    ```

- **Summary Command:**
    ```
    Enter a command: summary ./codefolder/mat_code.py
    Overall File Summary:
    This file contains mathematical functions such as factorial, power, and logarithm.

    Detailed Summaries:
    Function: factorial - This function calculates the factorial of a number.
    Function: power - This function calculates the power of a number.
    Function: logarithm - This function calculates the logarithm of a number.
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- This project uses [OpenAI API](https://platform.openai.com/docs/guides/embeddings) for generating embeddings and interacting with language models.
- [FAISS](https://github.com/facebookresearch/faiss) is used for the vector database.
- Special thanks to all the contributors and the open-source community.

---

Feel free to modify and enhance the `README.md` to suit your project's specific details and requirements.
