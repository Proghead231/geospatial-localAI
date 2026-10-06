import os
import json
import ollama
import asyncio
import nest_asyncio
from pypdf import PdfReader
from IPython.display import display, Markdown
import re
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
nest_asyncio.apply()

TRANSIENT_MARKERS = ("503", "429", "overloaded", "unavailable", "high demand",
                     "resource_exhausted", "timeout")


def _flatten_exceptions(exc):
    """Unwrap nested ExceptionGroups so the real cause becomes visible."""
    subs = getattr(exc, "exceptions", None)
    if subs:
        out = []
        for s in subs:
            out.extend(_flatten_exceptions(s))
        return out
    return [exc]

class ScientificAssistant:
    def __init__(self, model_name=None,
                mcp_command="poetry",
                mcp_args=("run", "python", "-m", "gee_mcp.server"),
                mcp_cwd="gee-mcp-main",
                env_vars=None):
        self.model_name = model_name
        self.mcp_command = mcp_command
        self.mcp_args = list(mcp_args)
        self.mcp_cwd = mcp_cwd
        self.env_vars = env_vars or {}
        self.messages = []
        self.loaded_docs = {}
        self._injected_docs = set()
        self.history_path = r"outputs/chat_history/chat_history.json"
        self.supported_exts = ('.py', '.txt', '.ipynb', '.pdf')
        self.tools = self._define_tier1_tools()


    def _define_tier1_tools(self):
        """Defines the Tier 1 scientific validation tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_factuality_issues",
                    "description": "Analyze a Google Earth Engine Python script and extract what aspects or issues are making scientific or data assumptions either explicitly or implicitly and might require factual verification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "python_code": {"type": "string"}
                        },
                        "required": ["question", "python_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_factuality_issue",
                    "description": "Make an assessment of a factuality issue identified in a Google Earth Engine Python script that attempts to answer an Earth Observation question, suggesting code updates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "python_code": {"type": "string"},
                            "issue_title": {"type": "string"},
                            "issue_description": {"type": "string"},
                            "issue_facts": {"type": "string"},
                            "issue_question_for_expert": {"type": "string"}
                        },
                        "required": ["question", "python_code", "issue_title", "issue_description", "issue_facts", "issue_question_for_expert"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "identify_sensible_variables",
                    "description": "Identifies the variables and constants within the GEE Python code whose values might impact the final result when executing the code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "python_code": {"type": "string"},
                            "baseline_answer": {"type": "string"}
                        },
                        "required": ["question", "python_code", "baseline_answer"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sensitivity_analysis",
                    "description": "Performs a sensitivity analysis by changing values of identified variables and recording how the final result changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "python_code": {"type": "string"},
                            "baseline_answer": {"type": "string"}
                        },
                        "required": ["question", "python_code", "baseline_answer"]
                    }
                }
            }
        ]

    async def _async_execute_mcp_tool(self, tool_name, arguments,
                                  retries=3, init_timeout=90, call_timeout=180):
        env = os.environ.copy()
        env.update(self.env_vars)

        params = StdioServerParameters(
            command=self.mcp_command,
            args=self.mcp_args,
            cwd=self.mcp_cwd,
            env=env,
        )

        last_err = ""
        for attempt in range(1, retries + 1):
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments), timeout=call_timeout
                        )

                text = "\n".join(b.text for b in result.content if hasattr(b, "text")) \
                    or "(tool returned no text)"

                is_error = getattr(result, "is_error", getattr(result, "isError", False))
                if not is_error:
                    return text
                last_err = f"MCP tool returned an error: {text}"

            except Exception as e:
                causes = _flatten_exceptions(e)
                detail = " | ".join(f"{type(c).__name__}: {c}" for c in causes)
                last_err = f"MCP connection/startup error: {detail}"

            print(f"⚠️ Attempt {attempt}/{retries} failed: {last_err[:500]}")

            if attempt < retries and any(m in last_err.lower() for m in TRANSIENT_MARKERS):
                wait = 2 ** attempt * 2          # 4s, 8s, ...
                print(f"   Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                break

        return last_err

    def _execute_mcp_tool(self, tool_name, arguments):
        """Synchronous wrapper for the MCP execution to keep the chat method clean."""
        print(f"\n⚙️ Triggering MCP Tool: {tool_name}...")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return loop.run_until_complete(self._async_execute_mcp_tool(tool_name, arguments))
        else:
            return asyncio.run(self._async_execute_mcp_tool(tool_name, arguments))

    def clear_memory(self):
        """Resets the chat context history completely."""
        self.messages = []
        self.loaded_docs = {}
        self._injected_docs = set()
        print("🧠 Assistant memory cleared.")

    def save_memory(self, path=None):
        path = path or self.history_path
        if (d := os.path.dirname(path)):
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "messages": self.messages,
                "loaded_docs": self.loaded_docs,
                "injected_docs": sorted(self._injected_docs)      # ← NEW
            }, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        print(f"💾 Saved {len(self.messages)} messages and {len(self.loaded_docs)} documents to {path}")


    def load_memory(self, path=None):
        path = path or self.history_path
        if not os.path.exists(path):
            print(f"⚠️ No history file at {path}, starting fresh.")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both old format (list) and new format (dict)
        if isinstance(data, list):
            self.messages = data
            self.loaded_docs = {}
            self._injected_docs = set()  
            print(f"📂 Loaded {len(self.messages)} messages (legacy format, no documents)")
        else:
            self.messages = data.get("messages", [])
            self.loaded_docs = data.get("loaded_docs", {})
            self._injected_docs = set(data.get("injected_docs", []))
            print(f"📂 Loaded {len(self.messages)} messages and {len(self.loaded_docs)} documents from {path}")

    def _parse_file(self, path):
        """Loads text from .py, .txt, .ipynb, or .pdf."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.py', '.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.ipynb':
            with open(path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            return "\n\n# --- Notebook Cell ---\n".join(
                "".join(c.get('source', []))
                for c in notebook.get('cells', [])
                if c.get('cell_type') == 'code'
            )
        elif ext == '.pdf':
            reader = PdfReader(path)
            return "\n".join(page.extract_text() for page in reader.pages)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _collect_target_files(self, folder_path=None, file_names=None, read_all=False):
        """
        Resolves target paths. Works with:
        1. folder_path + read_all=True
        2. folder_path + file_names
        3. Standalone file paths passed into file_names (without folder_path)
        """
        target_paths = []

        # Case 1: Read all supported files in a folder
        if read_all and folder_path:
            if not os.path.exists(folder_path):
                print(f"⚠️ Directory does not exist: {folder_path}")
                return []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(self.supported_exts):
                        target_paths.append(os.path.join(root, file))
            print(f"📁 Ingesting ALL ({len(target_paths)}) files from '{folder_path}'")

        # Case 2: Read specific file paths (with or without a folder_path prefix)
        elif file_names:
            if isinstance(file_names, str):
                file_names = [file_names]

            for name in file_names:
                # If folder_path is provided, combine them; otherwise treat 'name' as full path
                full_path = os.path.join(folder_path, name) if folder_path else name
                
                if os.path.exists(full_path):
                    target_paths.append(full_path)
                else:
                    print(f"⚠️ Specified file not found: {full_path}")

        return target_paths
    
    def remove_doc(self, filename):
        if filename in self.loaded_docs:
            del self.loaded_docs[filename]
            self._injected_docs.discard(filename)
            print(f"🗑️ Removed '{filename}' from loaded documents.")
        else:
            print(f"⚠️ '{filename}' not found in loaded documents.")

    def list_docs(self):
        if not self.loaded_docs:
            print("No documents currently loaded.")
        for name, content in self.loaded_docs.items():
            print(f"  📄 {name} ({len(content):,} chars)")
    
    @property
    def is_deepseek_r1(self):
        """Checks if the active model belongs to the DeepSeek-R1 family."""
        return bool(self.model_name and "deepseek-r1" in self.model_name.lower())

    def _trim_sliding_window(self, max_messages=20):
        """Keep messages[0] (system msg or R1 turn-1 blob) + the last N turns."""
        if len(self.messages) <= max_messages:
            return

        # Snapshot pre-trim state to a dated file via the existing save_memory
        folder = os.path.dirname(self.history_path) or "."
        stem = os.path.splitext(os.path.basename(self.history_path))[0]
        ext = os.path.splitext(self.history_path)[1] or ".json"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_path = os.path.join(folder, f"{stem}_{timestamp}{ext}")

        self.save_memory(path=archive_path)     # ← reuses existing function
        print(f"🗄️ Pre-trim snapshot saved → {archive_path}")

        anchor = self.messages[0]
        recent = self.messages[1:][-(max_messages - 1):]
        dropped = len(self.messages) - len(recent) - 1
        self.messages = [anchor] + recent
        print(f"✂️ Sliding window: dropped {dropped} old message(s), kept {len(self.messages)}.")

    def chat(self, prompt=None, folder_path=None, file_names=None, read_all=False,
     num_ctx=None, num_predict=None, num_batch=None, temperature=None, is_tool_response=False):
        """
        :param prompt: User question/instruction.
        :param folder_path: Target folder path (e.g., "./my_project" or "D:/research/papers").
        :param file_names: Single filename string or list of filenames (e.g., "model.py" or ["paper1.pdf", "script.py"]).
        :param read_all: If True, reads all supported files in folder_path.
        """
        if not is_tool_response:
            self._trim_sliding_window(max_messages=20)

            # Collect files based on inputs
            files_to_read = self._collect_target_files(folder_path, file_names, read_all)

            for file_path in files_to_read:
                name = os.path.basename(file_path)
                if name not in self.loaded_docs:
                    content = self._parse_file(file_path)
                    self.loaded_docs[name] = content
                    print(f"📄 Added '{name}' ({len(content)} characters)")
                else:
                    print(f"⏭️ '{name}' already loaded, skipping.")

            # Build context payload for all model types first
            context_payload = ""
            for name, content in self.loaded_docs.items():
                ext = os.path.splitext(name)[1].lower()
                if ext in ('.py', '.ipynb'):
                    context_payload += f"\n--- CODE FILE ({name}) ---\n```python\n{content}\n```\n"
                else:
                    context_payload += (
                        f"\n===== BEGIN FILE: {name} =====\n"
                        f"The following is the COMPLETE content of the file '{name}'. "
                        f"Read it directly; do not claim you cannot access it.\n\n"
                        f"{content[:30000]}\n"
                        f"===== END FILE: {name} =====\n"
                    )

        # Apply model-specific prompt structure
            if self.is_deepseek_r1:
                self.messages = [m for m in self.messages if m['role'] != 'system']
                has_prior_user_msgs = any(m['role'] == 'user' for m in self.messages)
                
                if not has_prior_user_msgs:
                    # Turn 1: Combine system, docs, and prompt
                    self.messages.append({'role': 'user', 'content': f"{context_payload}\n\n{prompt}"})
                    self._injected_docs = set(self.loaded_docs.keys())
                else:
                    # Subsequent turns: Just append the user's raw prompt
                    new_names = set(self.loaded_docs.keys()) - self._injected_docs
                    if new_names:
                        new_blob = ""
                        for name in new_names:
                            content = self.loaded_docs[name]
                            ext = os.path.splitext(name)[1].lower()
                            if ext in ('.py', '.ipynb'):
                                new_blob += f"\n--- NEW CODE FILE ({name}) ---\n```python\n{content}\n```\n"
                            else:
                                new_blob += f"\n--- NEW TEXT FILE ({name}) ---\n{content[:30000]}\n"
                        self.messages.append({'role': 'user', 'content': f"{new_blob}\n{prompt}"})
                        self._injected_docs |= new_names
                    else:
                        self.messages.append({'role': 'user', 'content': prompt})
                        
                    
            else:
                system_content = ""
                if context_payload:
                    system_content += f"\n\nLoaded documents for this session:\n{context_payload}"

                if self.messages and self.messages[0]['role'] == 'system':
                    self.messages[0]['content'] = system_content
                else:
                    self.messages.insert(0, {'role': 'system', 'content': system_content})

                self.messages.append({'role': 'user', 'content': prompt})

        current_tools = self.tools if not self.is_deepseek_r1 else None
        options = {
                "num_ctx":     num_ctx     if num_ctx     is not None else 8192,
                "num_predict": num_predict if num_predict is not None else 2048,
                "num_batch":   num_batch   if num_batch   is not None else 512,
                "temperature": temperature if temperature is not None else 0.6,
        }
        
        try:
            if not is_tool_response:
                print(f"🤖 ctx={options['num_ctx']} temp={options['temperature']} | Tools Active: {current_tools is not None}\n")

            if self.is_deepseek_r1:
                self.messages.append({'role': 'assistant', 'content': '<think>\n'})
                assistant_response = "<think>\n"
            else:
                assistant_response = ""
            
            def format_for_jupyter(text):
                # Collapse multiple consecutive <think> tags into a single one
                cleaned = re.sub(r'(<think>\s*)+', '<think>\n', text)
                return cleaned.replace("<think>", "### 🧠 Thought Process\n").replace("</think>", "\n\n---\n### 💡 Final Response\n")
            
            stream = ollama.chat(
                model=self.model_name, 
                messages=self.messages, 
                stream=True,
                options=options,
                tools=current_tools
            )

            tool_calls = []
            display_handle = None
            for chunk in stream:
                if "tool_calls" in chunk["message"]:
                    tool_calls = chunk["message"]["tool_calls"]
                    break

                text_chunk = chunk['message'].get('content') or ''
                if text_chunk: # Only attempt display updates if there is actual text
                    assistant_response += text_chunk
                    # Instantiate display handle ONLY once text starts arriving
                    if display_handle is None:
                        display_handle = display(Markdown(format_for_jupyter(assistant_response)), display_id=True)
                    else:
                        display_handle.update(Markdown(format_for_jupyter(assistant_response)))      
            if tool_calls:
                # Convert Ollama ToolCall objects to plain dicts to prevent json.dump errors
                serializable_tool_calls = []
                for tool in tool_calls:
                    if hasattr(tool, 'model_dump'):
                        serializable_tool_calls.append(tool.model_dump())
                    elif hasattr(tool, 'dict'):
                        serializable_tool_calls.append(tool.dict())
                    elif isinstance(tool, dict):
                        serializable_tool_calls.append(tool)
                    else:
                        serializable_tool_calls.append({
                            'function': {
                                'name': getattr(tool.function, 'name', ''),
                                'arguments': getattr(tool.function, 'arguments', {})
                            }
                        })

                self.messages.append({'role': 'assistant', 'content': assistant_response, 'tool_calls': serializable_tool_calls})
                
                for tool in tool_calls:
                    if isinstance(tool, dict):
                        func_name = tool['function']['name']
                        args = tool['function']['arguments']
                    else:
                        func_name = tool.function.name
                        args = tool.function.arguments

                    tool_result = self._execute_mcp_tool(func_name, args)
                    
                    self.messages.append({
                        'role': 'tool',
                        'name': func_name,
                        'content': tool_result
                    })
                
                return self.chat(
                    prompt=None, num_ctx=num_ctx, num_predict=num_predict, 
                    num_batch=num_batch, temperature=temperature, is_tool_response=True
                )
            
            else:
                if self.is_deepseek_r1:
                    self.messages[-1]['content'] = re.sub(r'(<think>\s*)+', '<think>\n', assistant_response)
                else:
                    self.messages.append({'role': 'assistant', 'content': assistant_response})
                print("\n\n✅ Turn Complete.")
                self.save_memory()

        except Exception as e:
            print(f"\n❌ Local LLM Execution Error: {e}")
            if self.is_deepseek_r1 and self.messages and self.messages[-1]['role'] == 'assistant':
                self.messages.pop()  # Remove incomplete prefill on error
            self.save_memory()