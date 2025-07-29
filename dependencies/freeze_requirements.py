import subprocess
import re

from pathlib import Path


def parse_heavy_dependencies(dependencies: list[str]):
    """
    Parse the output of `pip freeze` to identify heavy dependencies.
    """
    # Filter for heavy dependencies (e.g., those with 'torch' or 'transformers')
    heavy_dependencies = [
        dep for dep in dependencies if "torch" in dep or "transformers" in dep
    ]

    return heavy_dependencies


def get_package_list() -> list[str]:
    """
    Gets package list from pip freeze (only frozen packages, using ==)
    """
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    dependencies = result.stdout.splitlines()
    dependencies = [
        # package name == version
        dep
        for dep in dependencies
        if re.fullmatch(r"[a-zA-Z0-9\-_]+==.*", dep)
    ]
    return dependencies


def freeze_dependencies():
    """
    Freeze the heavy dependencies to a requirements file.
    """
    dependencies = get_package_list()
    heavy_dependencies = parse_heavy_dependencies(dependencies)

    with open(Path(__file__).parent / "heavy_requirements.txt", "w") as f:
        for dep in heavy_dependencies:
            f.write(f"{dep}\n")
    print(
        f"Heavy dependencies have been frozen to '{Path(__file__).parent / 'heavy_requirements.txt'}'."
    )

    with open(Path(__file__).parent / "frozen_requirements.txt", "w") as f:
        for dep in dependencies:
            f.write(f"{dep}\n")
    print(
        f"All dependencies have been frozen to '{Path(__file__).parent / 'frozen_requirements.txt'}'."
    )


if __name__ == "__main__":
    freeze_dependencies()
