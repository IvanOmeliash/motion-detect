"""motion_detect: aerial moving-target detection from drone video.

Classical camera-motion compensation -> TinyUNet (3-frame temporal input)
-> blob extraction -> SORT tracker. Trained on masks derived from ground-truth
bounding boxes with hard-negative mining.
"""

__version__ = "0.1.0"
