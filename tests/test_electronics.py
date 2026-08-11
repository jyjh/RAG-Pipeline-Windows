from __future__ import annotations

import pytest

from src.electronics import analyze_circuit, compare_measurements, parse_circuit, parse_value


DIVIDER = """\
V1 vin 0 DC 12
R1 vin out 1k
R2 out 0 1k
"""


def test_parse_spice_and_engineering_suffixes():
    result = parse_circuit(DIVIDER)
    assert result["format"] == "spice"
    assert result["errors"] == []
    assert result["component_graph"][1]["value"] == 1000.0
    assert parse_value("12mA") == pytest.approx(0.012)
    assert parse_value("2.2kOhm") == pytest.approx(2200.0)
    assert parse_value("4.7kΩ") == pytest.approx(4700.0)


def test_parse_forgiving_simple_format_and_ground_alias():
    result = parse_circuit("Voltage V1: VIN -> GND, 5V\nResistor R1: VIN -> OUT, 1k\nR2: OUT -> GND, 2k")
    assert result["format"] == "simple"
    assert result["errors"] == []
    assert result["nodes"][0] == "0"


def test_parser_reports_duplicates_missing_values_and_ambiguous_switches():
    result = parse_circuit("V1 a 0 5\nV1 b 0 3\nR1 a b\nS1 b 0")
    assert any("Duplicate" in error for error in result["errors"])
    assert any("Missing value" in error for error in result["errors"])
    assert any("explicit OPEN or CLOSED" in error for error in result["errors"])


def test_resistor_divider_operating_point_currents_and_power():
    result = analyze_circuit(DIVIDER)
    analysis = result["analysis"]
    assert analysis["status"] == "solved"
    assert analysis["node_voltages"]["out"] == pytest.approx(6.0)
    assert analysis["branch_currents"]["R1"] == pytest.approx(0.006)
    assert analysis["component_power"]["R2"] == pytest.approx(0.036)


def test_parallel_current_source_and_closed_switch_are_solved():
    result = analyze_circuit("I1 0 out 2m\nR1 out 0 1k\nS1 sense out CLOSED\nR2 sense 0 1k")
    assert result["analysis"]["status"] == "solved"
    assert result["analysis"]["node_voltages"]["out"] == pytest.approx(1.0)
    assert result["analysis"]["node_voltages"]["sense"] == pytest.approx(1.0)


def test_open_switch_creates_floating_node_and_singular_warning():
    result = analyze_circuit("V1 vin 0 5\nS1 vin out OPEN\nR1 out spare 1k")
    assert set(result["analysis"]["floating_nodes"]) >= {"out", "spare"}
    assert result["analysis"]["status"] == "indeterminate"


def test_contradictory_ideal_sources_are_indeterminate():
    result = analyze_circuit("V1 n 0 5\nV2 n 0 12")
    assert result["analysis"]["status"] == "indeterminate"
    assert "singular or contradictory" in result["analysis"]["error"]


def test_unsupported_component_is_structural_and_marks_partial_result():
    result = analyze_circuit("V1 vin 0 5\nR1 vin out 1k\nR2 out 0 1k\nQ1 out vin 0")
    assert result["unsupported_components"] == ["Q1"]
    assert result["analysis"]["status"] == "partial"
    assert result["component_graph"][-1]["supported"] is False
    assert result["component_graph"][-1]["terminals"] == ["out", "vin", "0"]


def test_voltage_measurement_mismatch_and_ranked_hypotheses():
    result = compare_measurements(
        DIVIDER,
        [{"type": "voltage", "positive_node": "out", "negative_node": "0", "value": "5V"}],
    )
    measurement = result["measurements"][0]
    assert measurement["status"] == "mismatch"
    assert measurement["expected"] == pytest.approx(6.0)
    assert {item["component_id"] for item in result["ranked_hypotheses"]} >= {"R1", "R2"}


def test_measurement_tolerance_polarity_and_logic_preservation():
    result = compare_measurements(
        DIVIDER,
        [
            {"type": "voltage", "positive_node": "0", "negative_node": "out", "value": "-6.1V"},
            {"type": "logic", "value": "HIGH", "node": "out"},
        ],
    )
    assert result["measurements"][0]["status"] == "match"
    assert result["measurements"][1]["observed"] == "high"
    assert result["measurements"][1]["status"] == "indeterminate"


def test_power_off_resistance_and_continuity_deactivate_voltage_sources():
    result = compare_measurements(
        DIVIDER,
        [
            {"type": "resistance", "positive_node": "out", "negative_node": "0", "value": "500ohm"},
            {"type": "continuity", "positive_node": "vin", "negative_node": "0", "value": "beep"},
        ],
    )
    assert result["measurements"][0]["status"] == "match"
    assert result["measurements"][0]["expected"] == pytest.approx(500.0)
    assert result["measurements"][1]["status"] == "match"
    assert result["measurements"][1]["equivalent_resistance_ohms"] == 0.0


def test_advisory_scope_warning_is_configurable():
    result = analyze_circuit("V1 n 0 24\nR1 n 0 1k", max_live_dc_voltage=12)
    assert any("configured 12 V DC advisory scope" in warning for warning in result["warnings"])
