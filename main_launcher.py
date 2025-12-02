"""
LUPE Main Launcher

This is the main entry point for the LUPE analysis suite.
It provides a simple launcher GUI with two options:

1. Run LUPE - Opens the main behavior classification GUI
2. Run AMPS - Opens the LUPE-AMPS pain scale analysis GUI

Both GUIs can run simultaneously, and the launcher stays open.

Usage:
    python main_launcher.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
from pathlib import Path


class LupeLauncher:
    """Main launcher window for LUPE analysis tools."""

    def __init__(self):
        """Initialize the launcher GUI."""
        self.root = tk.Tk()
        self.root.title("LUPE Analysis Suite")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # Track launched GUI processes
        self.lupe_process = None
        self.amps_process = None

        self._create_widgets()

    def _create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="LUPE Analysis Suite",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Light Automated Pain Evaluator 2.0",
            font=('Arial', 11)
        )
        subtitle_label.pack(pady=(0, 30))

        # Buttons frame (centered)
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(expand=True)

        # ===== LUPE Button =====
        lupe_frame = ttk.Frame(buttons_frame)
        lupe_frame.pack(pady=15)

        self.lupe_button = tk.Button(
            lupe_frame,
            text="Run LUPE",
            command=self._launch_lupe,
            font=('Arial', 16, 'bold'),
            bg='#4CAF50',
            fg='white',
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.lupe_button.pack()

        lupe_desc = ttk.Label(
            lupe_frame,
            text="Behavior Classification & Feature Extraction",
            font=('Arial', 9),
            foreground='gray'
        )
        lupe_desc.pack(pady=(5, 0))

        # ===== AMPS Button =====
        amps_frame = ttk.Frame(buttons_frame)
        amps_frame.pack(pady=15)

        self.amps_button = tk.Button(
            amps_frame,
            text="Run AMPS",
            command=self._launch_amps,
            font=('Arial', 16, 'bold'),
            bg='#2196F3',
            fg='white',
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.amps_button.pack()

        amps_desc = ttk.Label(
            amps_frame,
            text="Pain Scale Analysis (PCA Projection)",
            font=('Arial', 9),
            foreground='gray'
        )
        amps_desc.pack(pady=(5, 0))

        # ===== Exit Button =====
        exit_button = ttk.Button(
            main_frame,
            text="Exit",
            command=self._on_exit
        )
        exit_button.pack(pady=(30, 0))

        # Status label at bottom
        self.status_label = ttk.Label(
            main_frame,
            text="",
            font=('Arial', 9),
            foreground='blue'
        )
        self.status_label.pack(side=tk.BOTTOM, pady=(10, 0))

    def _launch_lupe(self):
        """Launch the main LUPE classification GUI in a separate process."""
        # Check if LUPE is already running
        if self.lupe_process is not None and self.lupe_process.poll() is None:
            messagebox.showinfo(
                "Already Running",
                "LUPE classification GUI is already running."
            )
            return

        # Update status
        self.status_label.config(text="Launching LUPE Classification GUI...")
        self.root.update()

        # Launch in separate process
        try:
            # Get the path to the main_lupe_gui.py script
            script_path = Path(__file__).parent / "main_lupe_gui.py"

            # Validate script exists
            if not script_path.exists():
                raise FileNotFoundError(f"GUI script not found: {script_path}")

            # Launch as subprocess (without capturing output to allow GUI to show)
            # Use CREATE_NO_WINDOW flag on Windows to prevent console window
            import platform
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # IMPORTANT: Do NOT use PIPE for stdout/stderr without reading them!
            # Pipe buffers are limited (~64KB on Windows) and will cause the
            # subprocess to hang when the buffer fills up during long operations.
            # Using DEVNULL discards output but prevents blocking.
            self.lupe_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                startupinfo=startupinfo,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )

            # Give it a moment to start, then check if it failed immediately
            import time
            time.sleep(1.0)  # Increased from 0.5s to 1.0s for more reliable detection

            # Check if process failed immediately
            if self.lupe_process.poll() is not None:
                # Process already terminated
                error_msg = f"Process exited with code {self.lupe_process.returncode}"
                messagebox.showerror(
                    "Launch Error",
                    f"LUPE GUI failed to start:\n\n{error_msg}\n\n"
                    "Check the console or run main_lupe_gui.py directly for details."
                )
                self.status_label.config(text="Launch failed. See error message.")
                return

            self.status_label.config(text="LUPE Classification GUI opened.")
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch LUPE GUI:\n{str(e)}"
            )
            self.status_label.config(text="")

    def _launch_amps(self):
        """Launch the LUPE-AMPS GUI in a separate process."""
        # Check if AMPS is already running
        if self.amps_process is not None and self.amps_process.poll() is None:
            messagebox.showinfo(
                "Already Running",
                "LUPE-AMPS analysis GUI is already running."
            )
            return

        # Update status
        self.status_label.config(text="Launching LUPE-AMPS GUI...")
        self.root.update()

        # Launch in separate process
        try:
            # Get the path to the main_lupe_amps_gui.py script
            script_path = Path(__file__).parent / "main_lupe_amps_gui.py"

            # Validate script exists
            if not script_path.exists():
                raise FileNotFoundError(f"GUI script not found: {script_path}")

            # Launch as subprocess (without capturing output to allow GUI to show)
            # Use CREATE_NO_WINDOW flag on Windows to prevent console window
            import platform
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # IMPORTANT: Do NOT use PIPE for stdout/stderr without reading them!
            # Pipe buffers are limited (~64KB on Windows) and will cause the
            # subprocess to hang when the buffer fills up during long operations.
            # Using DEVNULL discards output but prevents blocking.
            self.amps_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                startupinfo=startupinfo,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )

            # Give it a moment to start, then check if it failed immediately
            import time
            time.sleep(1.0)  # Increased from 0.5s to 1.0s for more reliable detection

            # Check if process failed immediately
            if self.amps_process.poll() is not None:
                # Process already terminated
                error_msg = f"Process exited with code {self.amps_process.returncode}"
                messagebox.showerror(
                    "Launch Error",
                    f"LUPE-AMPS GUI failed to start:\n\n{error_msg}\n\n"
                    "Check the console or run main_lupe_amps_gui.py directly for details."
                )
                self.status_label.config(text="Launch failed. See error message.")
                return

            self.status_label.config(text="LUPE-AMPS GUI opened.")
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch LUPE-AMPS GUI:\n{str(e)}"
            )
            self.status_label.config(text="")

    def _on_exit(self):
        """Handle exit button click."""
        # Check if any GUI processes are still running
        lupe_running = self.lupe_process is not None and self.lupe_process.poll() is None
        amps_running = self.amps_process is not None and self.amps_process.poll() is None

        if lupe_running or amps_running:
            response = messagebox.askyesno(
                "Confirm Exit",
                "One or more analysis GUIs are still running.\n"
                "The GUIs will continue running independently.\n\n"
                "Close the launcher?"
            )
            if not response:
                return

        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the launcher GUI main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the LUPE launcher."""
    print("=" * 60)
    print("LUPE Analysis Suite")
    print("=" * 60)
    print("\nLaunching main menu...\n")

    try:
        launcher = LupeLauncher()
        launcher.run()
    except Exception as e:
        print(f"\nError launching LUPE suite: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
