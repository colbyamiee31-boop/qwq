#!/usr/bin/env python3
"""
Unit test for JSON serialization compatibility with NumPy scalar and array types.
Verifies that finite_json / safe_json converts all NumPy types (np.bool_, np.integer,
np.floating, np.ndarray, non-finite values) to native Python equivalents before
calling json.dumps with allow_nan=False.
"""

import json
import numpy as np


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [safe_json(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.generic):
        return safe_json(value.item())
    return value


def test_numpy_json_serialization():
    payload = {
        "bool_true": np.bool_(True),
        "bool_false": np.bool_(False),
        "int8": np.int8(8),
        "int16": np.int16(16),
        "int32": np.int32(32),
        "int64": np.int64(64),
        "uint8": np.uint8(80),
        "float32": np.float32(3.14159),
        "float64": np.float64(2.718281828459045),
        "nan": np.float64(np.nan),
        "inf_pos": np.float64(np.inf),
        "inf_neg": np.float64(-np.inf),
        "array_1d": np.array([1, 2, 3]),
        "array_2d": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nested_dict": {
            "flag": np.bool_(True),
            "count": np.int64(97),
            "errors": [np.float64(1.23e-6), np.float64(np.nan)],
            "matrix": np.array([[np.bool_(True), np.bool_(False)]]),
        },
    }

    # Must serialize without TypeError or ValueError (allow_nan=False)
    cleaned = safe_json(payload)
    serialized = json.dumps(cleaned, indent=2, allow_nan=False)
    assert serialized is not None
    loaded = json.loads(serialized)

    assert loaded["bool_true"] is True
    assert loaded["bool_false"] is False
    assert loaded["int64"] == 64
    assert abs(loaded["float64"] - 2.718281828459045) < 1e-12
    assert loaded["nan"] is None
    assert loaded["inf_pos"] is None
    assert loaded["inf_neg"] is None
    assert loaded["array_1d"] == [1, 2, 3]
    assert loaded["nested_dict"]["flag"] is True
    assert loaded["nested_dict"]["count"] == 97
    assert loaded["nested_dict"]["errors"] == [1.23e-6, None]

    print("[PASS] test_json_numpy passed all assertions.")


if __name__ == "__main__":
    test_numpy_json_serialization()
