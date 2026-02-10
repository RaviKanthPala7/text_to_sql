"""
Entry point for the Text-to-SQL project.

By default this script runs the single-question example flow.
Other flows are available under the `scripts/` directory.
"""

import warnings 

from scripts.run_example import main as run_example_main


def main() -> None:
    warnings.filterwarnings("ignore")
    run_example_main()


if __name__ == "__main__":
    main()

