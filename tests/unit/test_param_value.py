"""Unit tests for tool parameter conversion."""

from parsers.tools import convert_param_value


def test_number_keeps_an_integer_past_the_float_mantissa() -> None:
    assert convert_param_value("9007199254740993", "number") == 9007199254740993
    assert convert_param_value("-9007199254740993", "number") == -9007199254740993
    assert convert_param_value("9007199254740993.0", "float") == 9007199254740993


def test_number_keeps_ordinary_values() -> None:
    assert convert_param_value("1.5", "number") == 1.5
    assert convert_param_value("2.0", "float") == 2
    assert convert_param_value("1e2", "number") == 100
    assert convert_param_value("42", "integer") == 42
