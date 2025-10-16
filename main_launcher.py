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
import threading
from pathlib import Path


class LupeLauncher:
    """Main launcher window for LUPE analysis tools."""

    def __init__(self):
        """Initialize the launcher GUI."""
        self.root = tk.Tk()
        self.root.title("LUPE Analysis Suite")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # Track launched GUIs to prevent multiple instances
        self.lupe_running = False
        self.amps_running = False

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
        """Launch the main LUPE classification GUI in a separate thread."""
        if self.lupe_running:
            messagebox.showinfo(
                "Already Running",
                "LUPE classification GUI is already running."
            )
            return

        # Update status
        self.status_label.config(text="Launching LUPE Classification GUI...")
        self.root.update()

        # Launch in separate thread
        thread = threading.Thread(target=self._run_lupe_gui, daemon=True)
        thread.start()

    def _launch_amps(self):
        """Launch the LUPE-AMPS GUI in a separate thread."""
        if self.amps_running:
            messagebox.showinfo(
                "Already Running",
                "LUPE-AMPS analysis GUI is already running."
            )
            return

        # Update status
        self.status_label.config(text="Launching LUPE-AMPS GUI...")
        self.root.update()

        # Launch in separate thread
        thread = threading.Thread(target=self._run_amps_gui, daemon=True)
        thread.start()

    def _run_lupe_gui(self):
        """Run the LUPE classification GUI (threaded)."""
        try:
            self.lupe_running = True

            # Import here to avoid circular imports and delayed loading
            from src.gui.main_window import LupeGUI

            # Create and run the GUI
            app = LupeGUI()

            # Update status
            self.status_label.config(text="LUPE Classification GUI opened.")

            # Run the GUI (this blocks until window closes)
            app.run()

        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch LUPE GUI:\n{str(e)}"
            )
        finally:
            self.lupe_running = False
            self.status_label.config(text="")

    def _run_amps_gui(self):
        """Run the LUPE-AMPS GUI (threaded)."""
        try:
            self.amps_running = True

            # Import here to avoid circular imports and delayed loading
            from src.gui.lupe_amps_window import LupeAmpsGUI

            # Create and run the GUI
            app = LupeAmpsGUI()

            # Update status
            self.status_label.config(text="LUPE-AMPS GUI opened.")

            # Run the GUI (this blocks until window closes)
            app.run()

        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch LUPE-AMPS GUI:\n{str(e)}"
            )
        finally:
            self.amps_running = False
            self.status_label.config(text="")

    def _on_exit(self):
        """Handle exit button click."""
        # Check if any GUIs are running
        if self.lupe_running or self.amps_running:
            response = messagebox.askyesno(
                "Confirm Exit",
                "One or more analysis GUIs are still running.\n"
                "Close the launcher anyway?"
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
