import subprocess
import os
import tempfile

class LeanVerifier:
    def __init__(self, project_path):
        self.project_path = project_path
        self.env = os.environ.copy()
        # Ensure proxies and cache are set correctly
        self.env["XDG_CACHE_HOME"] = r"D:\code\MPI\.cache"
        self.env["all_proxy"] = "socks5h://127.0.0.1:7890"
        self.env["http_proxy"] = "socks5h://127.0.0.1:7890"
        self.env["https_proxy"] = "socks5h://127.0.0.1:7890"

    def verify(self, lean_code):
        """
        Verifies the given Lean code by compiling it with lake env lean.
        Returns: (success: bool, output: str)
        """
        # Create a temporary file in the project directory
        # We use a fixed name to avoid polluting the directory, but could use tempfile
        filename = "TempSolution.lean"
        filepath = os.path.join(self.project_path, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(lean_code)
            
            # Run lake env lean
            # Note: We use shell=True on Windows for environment variables to persist properly in some cases,
            # but subprocess.run with env parameter is better.
            
            # Construct command
            # We assume 'lake' is in the PATH. If not, we might need absolute path.
            # Given previous tool usage, 'lake' seems available.
            
            result = subprocess.run(
                ["lake", "env", "lean", filename],
                cwd=self.project_path,
                env=self.env,
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            
            if result.returncode == 0:
                return True, "Verification Successful"
            else:
                return False, result.stderr + result.stdout

        except Exception as e:
            return False, str(e)
        finally:
            # Clean up
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

if __name__ == "__main__":
    # Test
    verifier = LeanVerifier(r"D:\code\MPI\competitions\AIMO\lean_solver")
    code_pass = "import Mathlib.Data.Nat.Basic\nexample : 1 + 1 = 2 := by rfl"
    print("Testing Pass:", verifier.verify(code_pass))
    
    code_fail = "import Mathlib.Data.Nat.Basic\nexample : 1 + 1 = 3 := by rfl"
    print("Testing Fail:", verifier.verify(code_fail))
