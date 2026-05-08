# -*- coding: utf-8 -*-
"""
src.data package

Data preprocessing and vocabulary building modules.

Modules:
    - init_data: Raw data cleaning, label generation, dataset splitting
    - preprocess: Chinese tokenization, vocabulary building, text serialization
"""

from .preprocess import TextPreprocessor
from .init_data import process_data

__all__ = ["TextPreprocessor", "process_data"]
