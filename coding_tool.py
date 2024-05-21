import openai
import faiss
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import re
import ast
import tokenize
from io import BytesIO

# Set up OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the function to generate embeddings
def get_embeddings(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = np.array(response['data'][0]['embedding'])
    return embedding

# Function to extract information using AST
def extract_ast_info(file_content):
    tree = ast.parse(file_content)
    elements = []
    metadata = {
        "num_classes": 0,
        "num_functions": 0,
        "file_path": "",
        "comments": []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            metadata["num_classes"] += 1
            class_info = {
                "type": "class",
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "methods": [],
                "start_line": node.lineno,
                "end_line": max(child.lineno for child in ast.walk(node) if hasattr(child, 'lineno'))
            }
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    method_info = {
                        "type": "method",
                        "name": child.name,
                        "docstring": ast.get_docstring(child),
                        "args": [arg.arg for arg in child.args.args],
                        "start_line": child.lineno,
                        "end_line": max(grandchild.lineno for grandchild in ast.walk(child) if hasattr(grandchild, 'lineno'))
                    }
                    class_info["methods"].append(method_info)
            elements.append(class_info)
        elif isinstance(node, ast.FunctionDef):
            metadata["num_functions"] += 1
            function_info = {
                "type": "function",
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "args": [arg.arg for arg in node.args.args],
                "start_line": node.lineno,
                "end_line": max(child.lineno for child in ast.walk(node) if hasattr(child, 'lineno'))
            }
            elements.append(function_info)
    
    return elements, metadata

# Function to extract comments from the file content
def extract_comments(file_content):
    comments = []
    tokens = tokenize.tokenize(BytesIO(file_content.encode('utf-8')).readline)
    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.COMMENT:
            comments.append(tokval)
    return comments

# Step 1: Build the VDB
print("Building Vector Database (VDB)...")
index = faiss.IndexFlatL2(1536)  # Assuming embedding size of 1536
file_index = {}

code_base_path = "./codefolder"  # Path to the code base
chunk_counter = 0

for root, dirs, files in os.walk(code_base_path):
    for file in tqdm(files, desc="Processing files for VDB"):
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                code = f.read()
                elements, metadata = extract_ast_info(code)
                metadata["file_path"] = file_path
                metadata["comments"] = extract_comments(code)
                metadata_text = json.dumps(metadata)
                metadata_embedding = get_embeddings(metadata_text)
                index.add(np.expand_dims(metadata_embedding, axis=0))
                file_index[chunk_counter] = {"file_path": file_path, "metadata": metadata}
                chunk_counter += 1

                for element in elements:
                    element_text = json.dumps(element)
                    embedding = get_embeddings(element_text)
                    index.add(np.expand_dims(embedding, axis=0))
                    file_index[chunk_counter] = {"file_path": file_path, "element": element, "metadata": metadata}
                    chunk_counter += 1

faiss.write_index(index, "code_embeddings.index")
with open("file_index.json", 'w') as f:
    json.dump(file_index, f)

# Step 2: Summarization
print("Summarizing code files...")
for root, dirs, files in os.walk(code_base_path):
    for file in tqdm(files, desc="Summarizing files"):
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                code = f.read()
                elements, _ = extract_ast_info(code)

                # Summarize the entire file first
                file_summary_response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Summarize the overall content of the following Python file:\n\n{code}"}
                    ],
                    max_tokens=100
                )
                file_summary = file_summary_response['choices'][0]['message']['content'].strip()
                
                # Summarize each element separately
                element_summaries = []
                for element in elements:
                    element_text = json.dumps(element)
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Summarize the following Python code element:\n\n{element_text}"}
                        ],
                        max_tokens=150
                    )
                    summary = response['choices'][0]['message']['content'].strip()
                    element_summaries.append(summary)
                
                # Combine file summary and element summaries
                combined_summary = f"Overall File Summary:\n{file_summary}\n\nDetailed Summaries:\n" + "\n".join(element_summaries)
                summary_file_path = os.path.join('summaries', f"summary_{file.replace('.py', '').replace(os.sep, '_')}.json")
                with open(summary_file_path, 'w') as summary_file:
                    json.dump({"summary": combined_summary}, summary_file)

# Step 3: Simple CLI with RAG Implementation
def query_codebase(question, k=5, threshold=1.0):
  
    # Handle general queries
    question_embedding = get_embeddings(question)
    distances, indices = index.search(np.expand_dims(question_embedding, axis=0), k=k)
    
    relevant_files = []
    for i, dist in zip(indices[0], distances[0]):
        if i in file_index and dist < threshold:
            file_data = file_index[i]
            file_path = file_data['file_path']
            function_contents = extract_ast_info(open(file_path, 'r').read())[0] # Extract AST elements
            file_data['function_contents'] = function_contents
            relevant_files.append((file_data, dist))

    return sorted(relevant_files, key=lambda x: x[1])

# Function to augment retrieved information and generate a response
def augment_and_generate(question, retrieved_info):
    context = "\n\n".join([json.dumps(info, indent=2) for info in retrieved_info])
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in understanding and analyzing Python code."},
            {"role": "user", "content": (
                f"Using the following context, which includes docstrings and comments, answer the question. "
                f"Understand the code structure, functions, classes, and their relationships."
                f"Read the code snippets and provide insights based on the context."
                f"Read Docstrings, comments, and code structure to provide a detailed response."
                f"Make sure to extract relevant details, provide examples if available, "
                f"and explain your reasoning step-by-step. Consider all functions, classes, comments, and their docstrings:\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                f"Please include code references, function names, class names, or specific lines if applicable, "
                f"and ensure the response is concise and clear."
            )}
        ],
        max_tokens=400
    )
    return response['choices'][0]['message']['content'].strip()

def log_query_results(query, results, response):
    # Convert results to a pretty string format
    log_entry = f"Query: {query}\n\n"
    log_entry += "Results:\n"
    for result, dist in results:
        log_entry += f"File: {result['file_path']}\n"
        if "element" in result:
            element = result["element"]
            log_entry += f"  Function: {element['name']}\n"
            log_entry += f"  Docstring: {element['docstring'] or 'No docstring available'}\n"
            log_entry += f"  Start Line: {element['start_line']}\n"
            log_entry += f"  End Line: {element['end_line']}\n"
        else:
            metadata = result["metadata"]
            log_entry += f"  Number of Functions: {metadata['num_functions']}\n"
            log_entry += f"  Number of Classes: {metadata['num_classes']}\n"
        log_entry += f"  Distance: {dist}\n"
        log_entry += "-" * 40 + "\n"
    log_entry += f"\nGenerated Response:\n{response}\n"
    log_entry += "=" * 80 + "\n\n"

    log_file_path = "query_log.txt"

    # Write log entry to the log file
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_entry)

def get_summary(file_path):
    summary_file_path = os.path.join('summaries', f"summary_{os.path.basename(file_path).replace('.py', '').replace(os.sep, '_')}.json")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as f:
            summary = json.load(f)
            return summary.get("summary", "No summary available.")
    return None

def get_ast(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
        elements, _ = extract_ast_info(code)
        return json.dumps(elements, indent=2)

def suggest_changes(file_path, prompt):
    with open(file_path, 'r') as f:
        code = f.read()
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the code:\n\n{code}\n\n{prompt}"}
            ],
            max_tokens=250
        )
        suggestion = response['choices'][0]['message']['content'].strip()
        return suggestion

def apply_changes(file_path, changes):
    try:
        with open(file_path, 'a') as f:
            f.write('\n' + changes)
        return True
    except Exception as e:
        print(f"Error applying changes: {e}")
        return False

if __name__ == "__main__":
    while True:
        print("\nAvailable commands:\n1. query <question>\n2. summary <file_path>\n3. ast <file_path>\n4. suggest <file_path> <prompt>\n5. apply <file_path> <changes>\n6. exit")

        command = input("Enter a command: ").strip().split(" ", 1)
        cmd = command[0].lower()

        if cmd == "exit":
            break
        
        elif cmd == "query":
            if len(command) < 2:
                print("Please provide a question for the query.")
                continue
            question = command[1]
            results = query_codebase(question, k=5)
            if results:
                if isinstance(results, str):  # GPT-4-turbo metadata query result
                    print(results)
                else:  # Regular query results
                    print("Relevant files and code snippets:")
                    for file_data, dist in results:
                        file_path = file_data['file_path']
                        comments = file_data['metadata'].get('comments', [])
                        if "element" in file_data:
                            element = file_data["element"]
                            name = element['name']
                            docstring = element['docstring'] or "No docstring available"
                            start_line = element['start_line']
                            end_line = element['end_line']
                            print(f"File: {file_path}\nFunction: {name}\nDocstring: {docstring}\nComments: {comments}\nLines: {start_line}-{end_line}\nDistance: {dist}\n{'-'*40}")
                        else:
                            metadata = file_data["metadata"]
                            print(f"File: {file_path}\nNumber of Functions: {metadata['num_functions']}\nNumber of Classes: {metadata['num_classes']}\nComments: {comments}\n{'-'*40}")
                    
                    # Augment and generate a response based on retrieved information
                    generated_response = augment_and_generate(question, [res[0] for res in results])
                    print(f"Generated Response: {generated_response}")
                    
                    # Log the query, results, and response
                    log_query_results(question, results, generated_response)
            else:
                print("No relevant files found.")
        
        elif cmd == "summary":
            if len(command) < 2:
                print("Please provide the file path for the summary.")
                continue
            file_path = command[1]
            summary = get_summary(file_path)
            if summary:
                print(summary)
            else:
                print(f"No summary found for {file_path}")
        
        elif cmd == "ast":
            if len(command) < 2:
                print("Please provide the file path for the AST.")
                continue
            file_path = command[1]
            ast_info = get_ast(file_path)
            if ast_info:
                print(ast_info)
            else:
                print(f"No AST found for {file_path}")
        
        elif cmd == "suggest":
            if len(command) < 2:
                print("Please provide the file path and the prompt for the suggestion.")
                continue
            try:
                file_path, prompt = command[1].split(' ', 1)
                suggestion = suggest_changes(file_path, prompt)
                if suggestion:
                    print("Suggested changes:")
                    print(suggestion)
                else:
                    print(f"No suggestions found for {file_path} with the given prompt.")
            except ValueError:
                print("Please provide both the file path and the prompt for the suggestion.")
        
        elif cmd == "apply":
            if len(command) < 2:
                print("Please provide the file path and the changes to apply.")
                continue
            try:
                file_path, changes = command[1].split(' ', 1)
                success = apply_changes(file_path, changes)
                if success:
                    print(f"Changes applied successfully to {file_path}")
                else:
                    print(f"Failed to apply changes to {file_path}")
            except ValueError:
                print("Please provide both the file path and the changes to apply.")
        
        else:
            print("Invalid command or missing arguments.")

