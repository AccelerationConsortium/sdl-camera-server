# Camera service development

This repository owns cameras only. It must never import or actuate robot APIs.
Use synthetic frames and fake drivers for automated tests. Physical camera tests
require explicit deployment/test authorization; never move equipment to test it.
Keep host addresses, serial mappings, tokens and images outside git. Public
examples use placeholders. Preserve the MIT attribution of extracted code.

Run `uv sync --extra test` and `uv run pytest`. Use one API process and one
worker owner per camera. Keep service credentials scoped to assigned cameras.
