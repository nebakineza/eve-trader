"""Compatibility wrapper.

Some environments/scripts expect `nexus/dashboard.py` to be the Streamlit entrypoint.
Our canonical dashboard is `nexus/command_center.py`.

This file intentionally delegates without changing UX.
"""

import runpy


if __name__ == "__main__":
    runpy.run_path("nexus/command_center.py", run_name="__main__")
