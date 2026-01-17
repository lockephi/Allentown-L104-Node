# [L104_CODE_SANDBOX] - Safe code execution environment
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

class CodeSandbox:
    """
    L104 Code Sandbox.
    Executes code safely with resource limits.
    """
    
    SUPPORTED_LANGUAGES = {
        "python": {
            "extension": ".py",
            "command": ["python3"],
            "timeout": 30
        },
        "javascript": {
            "extension": ".js",
            "command": ["node"],
            "timeout": 30
        },
        "bash": {
            "extension": ".sh",
            "command": ["bash"],
            "timeout": 15
        }
    }
    
    def __init__(self, workspace: str = "/workspaces/Allentown-L104-Node"):
        self.workspace = workspace
        self.sandbox_dir = os.path.join(workspace, ".sandbox")
        self.execution_history = []
        
        # Create sandbox directory
        os.makedirs(self.sandbox_dir, exist_ok=True)
    
    def execute(self, code: str, language: str = "python", 
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute code in sandbox and return results.
        """
        if language not in self.SUPPORTED_LANGUAGES:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "supported": list(self.SUPPORTED_LANGUAGES.keys())
            }
        
        lang_config = self.SUPPORTED_LANGUAGES[language]
        timeout = timeout or lang_config["timeout"]
        
        # Create temp file
        filename = f"sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}{lang_config['extension']}"
        filepath = os.path.join(self.sandbox_dir, filename)
        
        try:
            # Write code to file
            with open(filepath, 'w') as f:
                f.write(code)
            
            # Execute with timeout
            result = subprocess.run(
                lang_config["command"] + [filepath],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace,
                env={**os.environ, "PYTHONPATH": self.workspace}
            )
            
            output = result.stdout
            error = result.stderr
            exit_code = result.returncode
            
            execution_result = {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": output[:10000],
                "stderr": error[:5000],
                "language": language,
                "execution_time": datetime.now().isoformat()
            }
            
            # Log execution
            self.execution_history.append({
                "code_preview": code[:200],
                "result": execution_result,
                "timestamp": datetime.now().isoformat()
            })
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout}s",
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": language
            }
        finally:
            # Cleanup temp file
            try:
                os.remove(filepath)
            except:
                pass
    
    def execute_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Convenience method for Python execution."""
        return self.execute(code, "python", timeout)
    
    def execute_with_context(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code with pre-injected context variables.
        """
        context = context or {}
        
        # Build context injection code
        context_code = "# Injected context\n"
        for key, value in context.items():
            if isinstance(value, str):
                context_code += f'{key} = """{value}"""\n'
            else:
                context_code += f'{key} = {repr(value)}\n'
        context_code += "\n# User code\n"
        
        full_code = context_code + code
        return self.execute_python(full_code)
    
    def run_tests(self, test_code: str) -> Dict[str, Any]:
        """
        Run test code and parse results.
        """
        # Add test runner wrapper
        wrapped_code = f'''
import sys
import traceback

test_results = {{"passed": 0, "failed": 0, "errors": []}}

def test_assert(condition, message="Assertion failed"):
    global test_results
    if condition:
        test_results["passed"] += 1
        print(f"✓ {{message}}")
    else:
        test_results["failed"] += 1
        test_results["errors"].append(message)
        print(f"✗ {{message}}")

try:
{chr(10).join("    " + line for line in test_code.split(chr(10)))}
except Exception as e:
    test_results["failed"] += 1
    test_results["errors"].append(str(e))
    traceback.print_exc()

print()
print(f"Results: {{test_results['passed']}} passed, {{test_results['failed']}} failed")
'''
        
        return self.execute_python(wrapped_code)
    
    def generate_and_run(self, description: str) -> Dict[str, Any]:
        """
        Use AI to generate code from description, then execute it.
        """
        from l104_gemini_real import GeminiReal
        
        gemini = GeminiReal()
        if not gemini.connect():
            return {"success": False, "error": "Gemini not available"}
        
        prompt = f"""Generate Python code to: {description}

Requirements:
- Print all results
- Handle errors gracefully
- Keep it simple and focused
- No user input required

Return ONLY the Python code, no explanation."""

        code = gemini.generate(prompt)
        
        if not code:
            return {"success": False, "error": "Code generation failed"}
        
        # Clean up code (remove markdown if present)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Execute the generated code
        result = self.execute_python(code)
        result["generated_code"] = code
        
        return result
    
    def get_history(self, limit: int = 10) -> list:
        """Get recent execution history."""
        return self.execution_history[-limit:]
    
    def clear_sandbox(self):
        """Clear sandbox directory."""
        import shutil
        try:
            shutil.rmtree(self.sandbox_dir)
            os.makedirs(self.sandbox_dir, exist_ok=True)
            return True
        except:
            return False


# Singleton
code_sandbox = CodeSandbox()
