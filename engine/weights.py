"""
engine/weights.py

Load the HuggingFace GPT-2 checkpoint and convert it into this engine's module layout (name mapping, fused-QKV split, Conv1D -> Linear transpose, GELU-tanh).  |  Phase 1
"""

