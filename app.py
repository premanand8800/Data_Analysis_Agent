


# import streamlit as st
# import os
# import sys
# import subprocess
# import pandas as pd
# import io
# import traceback
# from contextlib import redirect_stdout
# from typing import List
# import logging
# import time

# # --- 1. Terminal Logging Configuration ---
# # This sets up logging to your terminal.
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - [%(levelname)s] - %(message)s",
#     stream=sys.stdout,  # Ensure logs go to standard output, visible in your terminal
# )
# logger = logging.getLogger(__name__)


# # --- 2. The Core "God Mode" Tool ---
# def execute_python_code(python_code: str, file_paths: List[str]) -> str:
#     """
#     Dynamically installs missing libraries and executes Python code against uploaded files.
#     This is a non-sandboxed tool with full local access.
#     """
#     logger.info("--- TOOL: Attempting to execute Python code ---")
#     logger.debug(f"Code to execute:\n{python_code}")
    
#     # Attempt to dynamically install missing libraries
#     try:
#         # A lightweight check to find ModuleNotFoundError
#         exec(compile(python_code, '<string>', 'exec'))
#     except (ModuleNotFoundError, ImportError) as e:
#         missing_library = e.name
#         logger.warning(f"Code requires missing library: '{missing_library}'. Attempting installation.")
#         st.warning(f"Attempting to install missing library: `{missing_library}`")
#         try:
#             result = subprocess.run(
#                 [sys.executable, "-m", "pip", "install", missing_library],
#                 check=True, capture_output=True, text=True
#             )
#             logger.info(f"Successfully installed '{missing_library}'.")
#             st.success(f"Successfully installed `{missing_library}`.")
#         except subprocess.CalledProcessError as install_error:
#             logger.error(f"Failed to install '{missing_library}'. PIP Error: {install_error.stderr}")
#             st.error(f"Failed to install `{missing_library}`.")
#             return f"ERROR: Failed to install dependency '{missing_library}'."

#     # Now, execute the code with full functionality
#     try:
#         dataframes = {os.path.basename(path): pd.read_csv(path) for path in file_paths}
#         logger.info(f"Loaded {len(dataframes)} files into 'dataframes' dictionary.")

#         local_scope = {"pd": pd, "dataframes": dataframes, "file_paths": file_paths, "st": st}
        
#         output_capture = io.StringIO()
#         with redirect_stdout(output_capture):
#             exec(python_code, globals(), local_scope)

#         printed_output = output_capture.getvalue()
#         final_result = local_scope.get("result")
        
#         response = "--- Execution Summary ---\n"
#         if printed_output:
#             logger.info(f"Captured stdout:\n{printed_output}")
#             response += f"**Console Output:**\n```\n{printed_output}\n```\n"
#         if final_result is not None:
#             logger.info(f"Captured result variable: {final_result}")
#             response += f"**Result Variable:**\n```\n{str(final_result)}\n```\n"
        
#         if not printed_output and final_result is None:
#             response += "Code executed without error, but produced no output or 'result' variable."
#             logger.info("Code executed successfully with no captured output.")

#         return response

#     except Exception:
#         error_trace = traceback.format_exc()
#         logger.error(f"An exception occurred during code execution:\n{error_trace}")
#         return f"**ERROR during code execution:**\n```\n{error_trace}\n```"


# # --- 3. Streamlit Application UI and Logic ---
# st.set_page_config(layout="wide", page_title="Local Code Interpreter")
# st.title("🤖 Local Code Interpreter")

# st.warning(
#     "**DANGER ZONE:** This AI can install packages and run any code on your computer. "
#     "For personal, local use only. Do not expose this app to the internet."
# )

# # Initialize session state variables
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "uploaded_file_paths" not in st.session_state:
#     st.session_state.uploaded_file_paths = []
# if "google_api_key" not in st.session_state:
#     st.session_state.google_api_key = None
# if "chat_session" not in st.session_state:
#     st.session_state.chat_session = None

# # Sidebar for configuration
# with st.sidebar:
#     st.header("1. Configuration")
#     api_key_input = st.text_input("Enter your Google API Key", type="password")

#     if api_key_input:
#         st.session_state.google_api_key = api_key_input

#     st.header("2. File Uploader")
#     uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

#     if uploaded_files:
#         temp_dir = "tmp_uploads"
#         os.makedirs(temp_dir, exist_ok=True)
#         st.session_state.uploaded_file_paths = []
#         for file in uploaded_files:
#             path = os.path.join(temp_dir, file.name)
#             with open(path, "wb") as f:
#                 f.write(file.getbuffer())
#             st.session_state.uploaded_file_paths.append(path)
#         logger.info(f"User uploaded {len(st.session_state.uploaded_file_paths)} file(s).")
#         st.success(f"Uploaded {len(st.session_state.uploaded_file_paths)} file(s).")

# # API Key Gatekeeper
# if not st.session_state.google_api_key:
#     st.info("Please enter your Google API Key in the sidebar to begin.")
#     st.stop()

# # Configure LLM only once a key is provided
# try:
#     import google.generativeai as genai
#     genai.configure(api_key=st.session_state.google_api_key)
#     model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
#     logger.info("Google AI SDK configured successfully.")
# except Exception as e:
#     logger.error(f"Failed to configure Google AI: {e}")
#     st.error(f"Failed to configure Google AI. Is your API key valid? Error: {e}")
#     st.stop()

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"], unsafe_allow_html=True)

# # Main chat input and agent loop
# if prompt := st.chat_input("Ask me to analyze your files..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # --- 4. Context-Aware Agent Logic ---
#     with st.chat_message("assistant"):
#         canvas = st.container() # The "canvas" for live updates
        
#         # Initialize chat session if it doesn't exist
#         if st.session_state.chat_session is None:
#             logger.info("No active chat session found. Initializing new session.")
#             # Context Engineering: Give the agent its core instructions
#             system_prompt = (
#                 "You are an expert Python data analyst agent. Your goal is to help the user by writing and executing Python code. "
#                 "You have a tool `execute_python_code` available. "
#                 "The user's uploaded files are pre-loaded into a `dataframes` dictionary. "
#                 "Analyze the user's request, write the necessary Python code, and reflect on the output. "
#                 "If you encounter an error, analyze the traceback and attempt to correct your code in the next step. "
#                 "When you have the final answer, clearly state it to the user. "
#                 "Always output your Python code in a single, clean block: ```python ... ```"
#             )
#             st.session_state.chat_session = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}])

#         # The self-correction loop
#         max_retries = 3
#         current_request = prompt
        
#         for i in range(max_retries):
#             canvas_content = f"**Loop {i+1}/{max_retries}:** Thinking..."
#             canvas.markdown(canvas_content)
#             logger.info(f"Agent Loop {i+1}/{max_retries}")

#             try:
#                 # Send the current request to the context-aware chat session
#                 logger.info(f"Sending prompt to Gemini: '{current_request}'")
#                 response = st.session_state.chat_session.send_message(current_request)
                
#                 # Extract code from the response
#                 code_to_run = response.text.strip().replace("```python", "").replace("```", "").strip()

#                 if not code_to_run:
#                     # If LLM just chats, show the chat and break the loop
#                     canvas_content += f"\n\n**Response:**\n{response.text}"
#                     canvas.markdown(canvas_content)
#                     break
                
#                 canvas_content += "\n\n**Generated Code:**"
#                 canvas.markdown(canvas_content)
#                 st.code(code_to_run, language="python")
                
#                 # Execute the code
#                 tool_output = execute_python_code(code_to_run, st.session_state.uploaded_file_paths)
                
#                 canvas_content += f"\n\n**Execution Result:**\n{tool_output}"
#                 canvas.markdown(tool_output)

#                 # Check for errors to decide if we need to loop again
#                 if "ERROR" in tool_output:
#                     logger.warning("Execution resulted in an error. Preparing for self-correction.")
#                     current_request = (
#                         "The previous code failed with the following error. "
#                         "Please analyze the error and the original request, then provide the corrected Python code. "
#                         f"Error:\n{tool_output}"
#                     )
#                     canvas_content += "\n\nAn error occurred. Agent will attempt to correct it."
#                     canvas.markdown("An error occurred. Agent will attempt to correct it.")
#                     time.sleep(1) # Pause for user to read
#                 else:
#                     logger.info("Code executed successfully. Ending loop.")
#                     canvas.success("Task completed successfully!")
                    
#                     # Check for and display plot
#                     if os.path.exists('output_plot.png'):
#                         st.image('output_plot.png')
#                         os.remove('output_plot.png') # Clean up
                    
#                     break # Exit loop on success
            
#             except Exception as e:
#                 logger.critical(f"A critical error occurred in the agent loop: {e}")
#                 st.error(f"A critical error occurred: {e}")
#                 break

#         else: # Runs if the for loop completes without a `break`
#             logger.error(f"Agent failed to resolve the task after {max_retries} retries.")
#             canvas.error("Agent could not complete the task. Please try rephrasing your request.")
            
#         # Append final assistant output to the UI history
#         final_assistant_message = canvas.markdown # The last state of the canvas
#         st.session_state.messages.append({"role": "assistant", "content": canvas_content})



# import streamlit as st
# import os
# import sys
# import subprocess
# import pandas as pd
# import io
# import traceback
# from contextlib import redirect_stdout
# from typing import List
# import logging
# import time
# import re

# # --- 1. Terminal Logging Configuration ---
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s", stream=sys.stdout)
# logger = logging.getLogger(__name__)

# # --- 2. Helper Function for Robust Code Extraction (Unchanged) ---
# def extract_python_code(text: str) -> str:
#     pattern = r"```python\n(.*?)\n```"
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         logger.info("Found a '```python' code block.")
#         return match.group(1).strip()
#     pattern = r"```\n(.*?)\n```"
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         logger.info("Found a '```' code block.")
#         return match.group(1).strip()
#     logger.warning("No complete code block was found in the LLM response.")
#     return ""

# # --- 3. The Core "God Mode" Tool (With Fix for Bug #1) ---
# def execute_python_code(python_code: str, file_paths: List[str]) -> str:
#     logger.info("--- TOOL: Attempting to execute Python code ---")
#     logger.debug(f"Code to execute:\n{python_code}")
    
#     # --- FIX FOR BUG #1: Pre-check for ModuleNotFoundError only ---
#     try:
#         # We compile the code to check for syntax errors and module import issues
#         compiled_code = compile(python_code, '<string>', 'exec')
#     except (ModuleNotFoundError, ImportError) as e:
#         missing_library = e.name
#         logger.warning(f"Code requires missing library: '{missing_library}'. Attempting installation.")
#         st.warning(f"Attempting to install missing library: `{missing_library}`")
#         try:
#             result = subprocess.run(
#                 [sys.executable, "-m", "pip", "install", missing_library],
#                 check=True, capture_output=True, text=True
#             )
#             logger.info(f"Successfully installed '{missing_library}'.")
#             st.success(f"Successfully installed `{missing_library}`.")
#             # Re-compile after successful installation
#             compiled_code = compile(python_code, '<string>', 'exec')
#         except subprocess.CalledProcessError as install_error:
#             error_msg = f"ERROR: Failed to install dependency '{missing_library}'."
#             logger.error(f"{error_msg} PIP Error: {install_error.stderr}")
#             st.error(f"Failed to install `{missing_library}`.")
#             return error_msg
#     except Exception as e:
#         # Catch other potential compilation errors
#         error_msg = f"ERROR: Code failed to compile. Syntax error?\n{e}"
#         logger.error(error_msg)
#         return error_msg
        
#     # Now, execute the compiled code
#     try:
#         dataframes = {os.path.basename(path): pd.read_csv(path) for path in file_paths}
#         logger.info(f"Loaded {len(dataframes)} files into 'dataframes' dictionary.")
#         local_scope = {"pd": pd, "dataframes": dataframes, "file_paths": file_paths, "st": st}
#         output_capture = io.StringIO()
#         with redirect_stdout(output_capture):
#             exec(compiled_code, globals(), local_scope)

#         printed_output = output_capture.getvalue()
#         final_result = local_scope.get("result")
        
#         response = "--- Execution Summary ---\n"
#         if printed_output:
#             response += f"**Console Output:**\n```\n{printed_output}\n```\n"
#         if final_result is not None:
#             response += f"**Result Variable:**\n```\n{str(final_result)}\n```\n"
#         if not printed_output and final_result is None:
#             response += "Code executed without error, but produced no output or 'result' variable."
#         return response
#     except Exception:
#         error_trace = traceback.format_exc()
#         logger.error(f"An exception occurred during code execution:\n{error_trace}")
#         return f"**ERROR during code execution:**\n```\n{error_trace}\n```"


# # --- 4. Streamlit Application UI and Logic (With Fix for Bug #2) ---
# st.set_page_config(layout="wide", page_title="Local Code Interpreter")
# st.title("🤖 Local Code Interpreter")
# st.warning("**DANGER ZONE:** This AI can run any code on your computer. For local use only.")

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "uploaded_file_paths" not in st.session_state:
#     st.session_state.uploaded_file_paths = []
# if "google_api_key" not in st.session_state:
#     st.session_state.google_api_key = None
# if "chat_session" not in st.session_state:
#     st.session_state.chat_session = None

# # Sidebar
# with st.sidebar:
#     st.header("1. Configuration")
#     api_key_input = st.text_input("Enter your Google API Key", type="password")
#     if api_key_input:
#         st.session_state.google_api_key = api_key_input
#     st.header("2. File Uploader")
#     uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
#     if uploaded_files:
#         temp_dir = "tmp_uploads"; os.makedirs(temp_dir, exist_ok=True)
#         st.session_state.uploaded_file_paths = []
#         for file in uploaded_files:
#             path = os.path.join(temp_dir, file.name); st.session_state.uploaded_file_paths.append(path)
#             with open(path, "wb") as f: f.write(file.getbuffer())
#         st.success(f"Uploaded {len(st.session_state.uploaded_file_paths)} file(s).")

# # Gatekeeper & LLM Config
# if not st.session_state.google_api_key:
#     st.info("Please enter your Google API Key to begin."); st.stop()
# try:
#     import google.generativeai as genai
#     genai.configure(api_key=st.session_state.google_api_key)
#     model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
# except Exception as e:
#     st.error(f"Failed to configure Google AI. Is your key valid? Error: {e}"); st.stop()

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"], unsafe_allow_html=True)

# # Main chat input and agent loop
# if prompt := st.chat_input("Ask me to analyze your files..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"): st.markdown(prompt)
    
#     with st.chat_message("assistant"):
#         canvas = st.container()
        
#         if st.session_state.chat_session is None:
#             system_prompt = (
#                 "You are an expert Python data analyst agent. Your goal is to help the user by writing and executing Python code. "
#                 "You must only respond with Python code inside a single ```python ... ``` block. "
#                 "The user's uploaded files are in a `dataframes` dictionary. If you need to know the filenames, "
#                 "use this code: `print(dataframes.keys())`."
#             )
#             st.session_state.chat_session = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}, {'role': 'model', 'parts': ["Understood. I will only respond with Python code."]}])

#         max_retries = 3
#         current_request = prompt
        
#         # --- FIX FOR BUG #2: Manually build the response string ---
#         final_assistant_message = ""

#         for i in range(max_retries):
#             loop_header = f"**Loop {i+1}/{max_retries}:** Thinking..."
#             canvas.markdown(loop_header)
#             final_assistant_message += loop_header + "\n\n"
            
#             try:
#                 response = st.session_state.chat_session.send_message(current_request)
#                 llm_response_text = response.text
#                 code_to_run = extract_python_code(llm_response_text)
                
#                 if not code_to_run:
#                     canvas.markdown(llm_response_text)
#                     final_assistant_message += llm_response_text
#                     break
                
#                 code_header = "**Generated Code:**"
#                 canvas.markdown(code_header)
#                 canvas.code(code_to_run, language="python")
#                 final_assistant_message += code_header + f"\n```python\n{code_to_run}\n```\n\n"

#                 tool_output = execute_python_code(code_to_run, st.session_state.uploaded_file_paths)
                
#                 result_header = "**Execution Result:**"
#                 canvas.markdown(result_header)
#                 canvas.markdown(tool_output)
#                 final_assistant_message += result_header + f"\n{tool_output}\n\n"

#                 if "ERROR" in tool_output:
#                     current_request = (
#                         "The previous code failed with this error. Provide the corrected Python code. "
#                         f"Error:\n{tool_output}"
#                     )
#                     error_notice = "An error occurred. Agent will attempt to correct it."
#                     canvas.markdown(error_notice)
#                     final_assistant_message += error_notice + "\n\n"
#                     time.sleep(1)
#                 else:
#                     success_notice = "Task completed successfully!"
#                     canvas.success(success_notice)
#                     final_assistant_message += success_notice
#                     if os.path.exists('output_plot.png'):
#                         canvas.image('output_plot.png'); os.remove('output_plot.png')
#                     break
            
#             except Exception as e:
#                 critical_error_msg = f"A critical error occurred in the agent loop: {traceback.format_exc()}"
#                 st.error(critical_error_msg); break
#         else:
#             fail_notice = "Agent could not complete the task. Please try rephrasing your request."
#             canvas.error(fail_notice)
#             final_assistant_message += fail_notice
            
#         st.session_state.messages.append({"role": "assistant", "content": final_assistant_message})






import streamlit as st
import os
import sys
import subprocess
import pandas as pd
import io
import traceback
from contextlib import redirect_stdout
from typing import List
import logging
import time
import re

# --- 1. Terminal Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- 2. Helper Function for Robust Code Extraction ---
def extract_python_code(text: str) -> str:
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        logger.info("Found a '```python' code block.")
        return match.group(1).strip()
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        logger.info("Found a '```' code block.")
        return match.group(1).strip()
    logger.warning("No complete code block was found in the LLM response.")
    return ""

# --- 3. The Core "God Mode" Tool ---
def execute_python_code(python_code: str, file_paths: List[str]) -> str:
    logger.info("--- TOOL: Attempting to execute Python code ---")
    logger.debug(f"Code to execute:\n{python_code}")
    
    try:
        compiled_code = compile(python_code, '<string>', 'exec')
    except (ModuleNotFoundError, ImportError) as e:
        missing_library = e.name
        st.warning(f"Attempting to install missing library: `{missing_library}`")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", missing_library], check=True, capture_output=True, text=True)
            st.success(f"Successfully installed `{missing_library}`.")
            compiled_code = compile(python_code, '<string>', 'exec')
        except subprocess.CalledProcessError as install_error:
            return f"ERROR: Failed to install dependency '{missing_library}'."
    except Exception as e:
        return f"ERROR: Code failed to compile. Syntax error?\n{e}"
        
    try:
        dataframes = {os.path.basename(path): pd.read_csv(path) for path in file_paths}
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        local_scope = {"pd": pd, "np": np, "plt": plt, "sns": sns, "dataframes": dataframes, "file_paths": file_paths}
        output_capture = io.StringIO()
        with redirect_stdout(output_capture):
            exec(compiled_code, globals(), local_scope)

        printed_output = output_capture.getvalue()
        final_result = local_scope.get("result")
        
        response = "--- Execution Summary ---\n"
        if printed_output: response += f"**Console Output:**\n```\n{printed_output}\n```\n"
        if final_result is not None: response += f"**Result Variable:**\n```\n{str(final_result)}\n```\n"
        if not printed_output and final_result is None: response += "Code executed, but produced no output or 'result' variable."
        return response
    except Exception:
        return f"**ERROR during code execution:**\n```\n{traceback.format_exc()}\n```"

# --- 4. Streamlit UI and Logic with Multi-File System Prompt ---
st.set_page_config(layout="wide", page_title="Local Code Interpreter")
st.title("🤖 Local Code Interpreter")
st.warning("**DANGER ZONE:** This AI can run any code on your computer. For local use only.")

# Initialize session state
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_file_paths" not in st.session_state: st.session_state.uploaded_file_paths = []
if "google_api_key" not in st.session_state: st.session_state.google_api_key = None
if "chat_session" not in st.session_state: st.session_state.chat_session = None

# Sidebar
with st.sidebar:
    st.header("1. Configuration")
    api_key_input = st.text_input("Enter your Google API Key", type="password")
    if api_key_input: st.session_state.google_api_key = api_key_input
    st.header("2. File Uploader")
    uploaded_files = st.file_uploader("Upload one or more CSV files", type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        temp_dir = "tmp_uploads"; os.makedirs(temp_dir, exist_ok=True)
        st.session_state.uploaded_file_paths = []
        for file in uploaded_files:
            path = os.path.join(temp_dir, file.name); st.session_state.uploaded_file_paths.append(path)
            with open(path, "wb") as f: f.write(file.getbuffer())
        st.success(f"Uploaded {len(st.session_state.uploaded_file_paths)} file(s).")

# Gatekeeper & LLM Config
if not st.session_state.google_api_key:
    st.info("Please enter your Google API Key to begin."); st.stop()
try:
    import google.generativeai as genai
    genai.configure(api_key=st.session_state.google_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except Exception as e:
    st.error(f"Failed to configure Google AI. Is your key valid? Error: {e}"); st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"], unsafe_allow_html=True)

# Main chat input and agent loop
if prompt := st.chat_input("Ask me to analyze your files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        canvas = st.container()
        
        if st.session_state.chat_session is None:
            logger.info("Initializing new chat session with MULTI-FILE system prompt.")
            
            # --- THE FINAL, SUPERCHARGED SYSTEM PROMPT ---
            system_prompt = (
                "You are an elite Python data analyst agent. Your primary goal is to help users by writing and executing "
                "Python code to analyze data and create visualizations. You must follow these rules strictly:\n\n"
                "1.  **Environment:** You have access to `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, and `seaborn as sns`.\n"
                "2.  **Multi-File Data Access:** The user can upload multiple CSV files. They are all loaded into a dictionary of pandas DataFrames called `dataframes`. "
                "The keys of the dictionary are the filenames (e.g., `'sales_q1.csv'`).\n"
                "   - **CRITICAL:** If the user's request is ambiguous about which file to use, your **first step must be** to list the available files for the user with `print(dataframes.keys())`. This allows the user to clarify.\n"
                "   - You can process all dataframes at once. For example, to combine them: `combined_df = pd.concat(dataframes.values(), ignore_index=True)`.\n"
                "3.  **Visualization:** To create a plot, you MUST save it to a file using `plt.savefig('output_plot.png')`. **NEVER use `plt.show()`**.\n"
                "4.  **Workflow:** Think in steps. First, inspect the data (`df.head()`, `df.info()`). Then, perform the analysis. Finally, present the result with `print()` or a plot.\n"
                "5.  **Output Format:** You MUST only respond with Python code inside a single ````python ... ```` block. Do not write any conversational text outside the code block."
            )
            
            st.session_state.chat_session = model.start_chat(history=[
                {'role': 'user', 'parts': [system_prompt]}, 
                {'role': 'model', 'parts': ["Understood. I am an elite Python data analyst. I will handle multiple files by first listing them if the user's request is unclear. I will only respond with Python code and will use `plt.savefig` for plots."]}
            ])

        max_retries = 3
        current_request = prompt
        final_assistant_message = ""

        for i in range(max_retries):
            loop_header = f"**Loop {i+1}/{max_retries}:** Thinking..."
            canvas.markdown(loop_header); final_assistant_message += loop_header + "\n\n"
            
            try:
                response = st.session_state.chat_session.send_message(current_request)
                llm_response_text = response.text
                code_to_run = extract_python_code(llm_response_text)
                
                if not code_to_run:
                    canvas.markdown(llm_response_text); final_assistant_message += llm_response_text
                    break
                
                code_header = "**Generated Code:**"
                canvas.markdown(code_header); canvas.code(code_to_run, language="python")
                final_assistant_message += code_header + f"\n```python\n{code_to_run}\n```\n\n"

                tool_output = execute_python_code(code_to_run, st.session_state.uploaded_file_paths)
                
                result_header = "**Execution Result:**"
                canvas.markdown(result_header); canvas.markdown(tool_output)
                final_assistant_message += result_header + f"\n{tool_output}\n\n"

                if "ERROR" in tool_output:
                    current_request = ("The previous code failed. Analyze the traceback and provide the corrected Python code. "
                                     f"Error:\n{tool_output}")
                    error_notice = "An error occurred. Agent will attempt to correct it."
                    canvas.markdown(error_notice); final_assistant_message += error_notice + "\n\n"
                    time.sleep(1)
                else:
                    success_notice = "Task completed successfully!"
                    canvas.success(success_notice); final_assistant_message += success_notice
                    if os.path.exists('output_plot.png'):
                        canvas.image('output_plot.png'); os.remove('output_plot.png')
                    break
            
            except Exception as e:
                critical_error_msg = f"A critical error occurred in the agent loop: {traceback.format_exc()}"
                st.error(critical_error_msg); break
        else:
            fail_notice = "Agent could not complete the task. Please try rephrasing your request."
            canvas.error(fail_notice); final_assistant_message += fail_notice
            
        st.session_state.messages.append({"role": "assistant", "content": final_assistant_message})