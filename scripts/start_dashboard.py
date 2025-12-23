"""
Start the Streamlit dashboard for Kalshi Trading Agent.

This script starts the dashboard server on port 8501.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start the Streamlit dashboard."""
    project_root = Path(__file__).parent.parent
    app_path = project_root / "gui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Dashboard app not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting Kalshi Trading Agent Dashboard")
    print("=" * 60)
    print(f"Dashboard will be available at: http://localhost:8501")
    print(f"Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.headless", "true",
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

