#!/usr/bin/env python3
"""Test pattern library and synthesis engine."""
import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from l104_asi.code_generation import CodeGenerationEngine
e = CodeGenerationEngine()
print(f"Patterns: {len(e.synthesizer.library.patterns)}")

tests = [
    # (description, func_name, signature)
    ("Find the zero of a polynomial given as list of coefficients", "find_zero", "def find_zero(xs: list):"),
    ("Check if any two elements in list are closer than threshold", "has_close_elements", "def has_close_elements(numbers: list, threshold: float) -> bool:"),
    ("Return minimum sum of contiguous subarray", "minSubArraySum", "def minSubArraySum(nums: list) -> int:"),
    ("Build histogram of word frequencies returning most frequent", "histogram", "def histogram(test: str) -> dict:"),
    ("Check if array can be sorted by one right rotation", "move_one_ball", "def move_one_ball(arr: list) -> bool:"),
    ("Filter strings by even length then sort by length", "sorted_list_sum", "def sorted_list_sum(lst: list) -> list:"),
    ("Check if any rotation of word is substring of string", "cycpattern_check", "def cycpattern_check(a: str, b: str) -> bool:"),
    ("Return product of signs times sum of absolute values", "prod_signs", "def prod_signs(arr: list):"),
    ("Compute cumulative sum of a list", "cumsum", "def cumsum(nums: list) -> list:"),
    ("Swap case of all letters in a string", "swap_case", "def swap_case(s: str) -> str:"),
    ("Count number of digits in a number", "count_digits", "def count_digits(n: int) -> int:"),
    ("Compute the product of all elements", "product", "def product(nums: list) -> int:"),
]

for desc, fname, sig in tests:
    r = e.synthesizer.generate(desc, fname, sig)
    method = r["method"]
    pattern = r.get("pattern_used", "-")
    valid = r["syntax_valid"]
    print(f"  {fname:25s}: method={method:15s} pattern={str(pattern):30s} valid={valid}")
