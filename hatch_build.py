"""Hatch build hook."""

import importlib.util
import sys
from pathlib import Path
from typing import Any, override

from hatchling.metadata.plugin.interface import MetadataHookInterface


class AboutMetadataHookOld(MetadataHookInterface):
    """Hatchling metadata hook that loads dynamic metadata for authors, and URLs from the `__about__.py` file within the project."""

    @override
    def update(self, metadata: dict[str, Any]) -> None:
        """
        Update the metadata dictionary with values from the `__about__.py` file.

        Args:
            metadata: The dictionary containing the project metadata.
        """
        # Dynamically load __about__.py
        about_path = Path(self.root) / "src/pythemize/__about__.py"
        spec = importlib.util.spec_from_file_location("__about__", about_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load metadata from {about_path}")  # noqa: TRY003

        about = importlib.util.module_from_spec(spec)
        sys.modules["__about__"] = about
        spec.loader.exec_module(about)

        # Map __about__ attributes to metadata fields
        metadata["authors"] = [{"name": about.__author__, "email": about.__author_email__}]
        metadata["urls"] = {
            "Repository": about.__repo_url__,
            "Issues": about.__issues_url__,
            "Documentation": about.__documentation__,
        }
