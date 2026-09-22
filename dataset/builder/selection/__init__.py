"""Public dataset selection functions and geometry classification."""

from .geometry import _intersection, _best_short_side, _size_tag, classify
from .diversity import _FarthestSelector, _identity, SELECTION_FIELDS, _RARE_CLASSES
from .initial import provisional_selection
from .supplement import supplemental_selection
