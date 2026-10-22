"""The fusion prompt used by the current experiment configurations.

Rendered wording and version identifiers are part of experiment provenance.
Change them only for a deliberately versioned prompt change.
"""

from . import typed_paco_v13

PROMPT_FAMILIES = {"typed_paco_v13": typed_paco_v13}
