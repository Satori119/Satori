import asyncio
import subprocess
import os
import time
import threading
import sys
from pathlib import Path
from typing import Optional
from satori.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class AutoUpdateService:

    def __init__(
        self,
        github_repo: str,
        branch: str = "main",
        check_interval: int = 300,
        restart_delay: int = 10
    ):
        self.github_repo = github_repo
        self.branch = branch
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        
        self.script_dir = self._detect_script_directory()
        self.neuron_type = self._detect_neuron_type()
        self.project_root = self._detect_project_root()
        
        self.current_commit: Optional[str] = None
        self.is_running = False
        self._check_thread: Optional[threading.Thread] = None
        self._update_in_progress = threading.Lock()
    
    def _detect_script_directory(self) -> Path:
        main_module = sys.modules.get('__main__')
        if main_module and hasattr(main_module, '__file__'):
            return Path(main_module.__file__).parent.absolute()
        
        return Path.cwd()
    
    def _detect_neuron_type(self) -> str:
        script_dir_str = str(self.script_dir)
        
        if 'task_center' in script_dir_str:
            return 'task_center'
        elif 'validator' in script_dir_str:
            return 'validator'
        elif 'miner' in script_dir_str:
            return 'miner'
        else:
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, '__file__'):
                script_name = Path(main_module.__file__).stem
                if 'task_center' in script_name:
                    return 'task_center'
                elif 'validator' in script_name:
                    return 'validator'
                elif 'miner' in script_name:
                    return 'miner'
            
            return 'task_center'
    
    def _detect_project_root(self) -> Path:
        current = self.script_dir
        
        while current != current.parent:
            if (current / 'setup.py').exists() or (current / 'satori').exists():
                return current
            current = current.parent
        
        return self.script_dir.parent.parent
    
    async def start(self):
        if self.is_running:
            logger.warning("Auto-update service is already running")
            return
        
        self.is_running = True
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="auto-updater"
        )
        self._check_thread.start()
        logger.info(f"Auto-update service started (checking every {self.check_interval}s, neuron_type={self.neuron_type})")
    
    async def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=5.0)
        logger.info("Auto-update service stopped")
    
    def _monitor_loop(self):
        while self.is_running:
            try:
                self._perform_update_check()
            except Exception as e:
                logger.error(f"Error in update check loop: {e}", exc_info=True)
            
            sleep_count = 0
            while self.is_running and sleep_count < self.check_interval:
                time.sleep(1)
                sleep_count += 1
    
    def _perform_update_check(self):
        if not self._update_in_progress.acquire(blocking=False):
            logger.debug("Update check skipped - update already in progress")
            return
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            
            try:
                current_branch = self._execute_git_command(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
                
                local_commit = self._execute_git_command(["rev-parse", "HEAD"]).strip()
                
                logger.debug("Fetching latest changes from remote...")
                self._execute_git_command(["fetch", "origin"], check=False)
                
                remote_ref = f"origin/{current_branch}"
                remote_commit = self._execute_git_command(["rev-parse", remote_ref]).strip()
                
                if self._needs_update(local_commit, remote_commit):
                    logger.info(f"Update detected: local={local_commit[:8]}, remote={remote_commit[:8]}")
                    self._apply_update(remote_commit, current_branch)
                else:
                    if self.current_commit is None:
                        self.current_commit = local_commit
                        logger.info(f"Initial commit: {local_commit[:8]}")
                    else:
                        logger.debug(f"Repository is up-to-date (commit: {local_commit[:8]})")
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Update check failed: {e}", exc_info=True)
        finally:
            self._update_in_progress.release()
    
    def _needs_update(self, local_commit: str, remote_commit: str) -> bool:
        return local_commit != remote_commit
    
    def _apply_update(self, target_commit: str, branch: str):
        logger.info(f"Applying update to commit {target_commit[:8]}...")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.script_dir)
            
            try:
                reset_cmd = ["git", "reset", "--hard", target_commit]
                self._execute_git_command(reset_cmd, capture_output=False, check=True)
                logger.info(f"Successfully updated to commit {target_commit[:8]}")
                
                self.current_commit = target_commit
                
                self._run_post_update_steps()
                
                logger.info(f"Scheduling restart in {self.restart_delay} seconds...")
                time.sleep(self.restart_delay)
                
                self._restart_service()
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Failed to apply update: {e}", exc_info=True)
            raise
    
    def _run_post_update_steps(self):
        logger.info("Running post-update steps...")
        
        try:
            setup_script = self.project_root / "scripts" / "setup_env.sh"
            
            if setup_script.exists():
                venv_paths = [
                    self.project_root.parent / "venv_bitcast",
                    self.project_root / "venv",
                    self.project_root.parent / "venv"
                ]
                
                venv_path = None
                for path in venv_paths:
                    if path.exists() and (path / "bin" / "activate").exists():
                        venv_path = path
                        break
                
                if venv_path:
                    activate_script = venv_path / "bin" / "activate"
                    setup_cmd = f"source {activate_script} && {setup_script}"
                    
                    logger.info(f"Executing setup script: {setup_script}")
                    process = subprocess.Popen(
                        setup_cmd,
                        shell=True,
                        executable='/bin/bash',
                        cwd=self.project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    stdout, stderr = process.communicate(timeout=300)
                    
                    if process.returncode == 0:
                        logger.info("Post-update setup completed successfully")
                    else:
                        logger.warning(f"Setup script returned non-zero exit code: {process.returncode}")
                        if stderr:
                            logger.warning(f"Setup script stderr: {stderr.decode('utf-8', errors='ignore')}")
                else:
                    logger.warning("No virtual environment found, skipping setup script")
            else:
                logger.debug("No setup script found, skipping post-update steps")
                
        except subprocess.TimeoutExpired:
            logger.error("Setup script execution timed out")
        except Exception as e:
            logger.warning(f"Post-update steps failed (non-critical): {e}")
    
    def _restart_service(self):
        logger.info("Initiating service restart...")
        
        try:
            start_script = self.project_root / "scripts" / f"run_{self.neuron_type}.sh"
            
            if start_script.exists():
                venv_paths = [
                    self.project_root.parent / "venv_bitcast",
                    self.project_root / "venv",
                    self.project_root.parent / "venv"
                ]
                
                venv_path = None
                for path in venv_paths:
                    if path.exists() and (path / "bin" / "activate").exists():
                        venv_path = path
                        break
                
                if venv_path:
                    activate_script = venv_path / "bin" / "activate"
                    start_cmd = f"source {activate_script} && {start_script}"
                    
                    logger.info(f"Executing start script: {start_script}")
                    subprocess.Popen(
                        start_cmd,
                        shell=True,
                        executable='/bin/bash',
                        cwd=self.project_root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
                    
                    time.sleep(2)
                    
                    logger.info("Exiting current process...")
                    sys.exit(0)
                else:
                    logger.warning("No virtual environment found, using direct Python restart")
                    self._restart_via_python()
            else:
                logger.info("No start script found, using direct Python restart")
                self._restart_via_python()
                
        except Exception as e:
            logger.error(f"Failed to restart via script: {e}, attempting direct Python restart")
            self._restart_via_python()
    
    def _restart_via_python(self):
        try:
            logger.info("Restarting via Python process re-execution...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart Python process: {e}")
            raise
    
    def _execute_git_command(self, args: list, capture_output: bool = True, check: bool = True) -> Optional[str]:
        cmd = ["git"] + args
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=check
                )
                if check or result.returncode == 0:
                    return result.stdout.strip() if result.stdout else ""
                return None
            else:
                result = subprocess.run(
                    cmd,
                    timeout=60,
                    check=check
                )
                return None
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}, error: {e}")
            if check:
                raise
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing git command: {e}")
            raise
