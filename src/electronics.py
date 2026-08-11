from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np


GROUND_ALIASES = {"0", "gnd", "ground"}
SUPPORTED_TYPES = {"resistor", "voltage_source", "current_source", "wire", "switch"}
VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(meg|[TGMkmunpf])?\s*(?:ohms?|[vVaA]|amps?|[ΩΩ])?\s*$",
    re.IGNORECASE,
)
SUFFIXES = {
    "": 1.0,
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


class CircuitError(ValueError):
    pass


@dataclass
class Component:
    id: str
    type: str
    positive_node: str
    negative_node: str
    value: float | None = None
    unit: str = ""
    state: str | None = None
    raw: str = ""
    extra_terminals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "terminals": [self.positive_node, self.negative_node, *self.extra_terminals],
            "value": self.value,
            "unit": self.unit,
            "state": self.state,
            "supported": self.type in SUPPORTED_TYPES,
            "raw": self.raw,
        }


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def parse_value(value: Any, *, quantity: str = "value") -> float:
    if isinstance(value, bool):
        raise CircuitError(f"Invalid {quantity}: {value!r}.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = VALUE_RE.match(str(value or ""))
        if not match:
            raise CircuitError(f"Invalid {quantity}: {value!r}.")
        suffix = (match.group(2) or "").lower()
        number = float(match.group(1)) * SUFFIXES[suffix]
    if not math.isfinite(number):
        raise CircuitError(f"Invalid {quantity}: value must be finite.")
    return number


def _node(value: str) -> str:
    cleaned = str(value).strip().strip(",")
    if not cleaned:
        raise CircuitError("A component terminal is missing a node name.")
    return "0" if cleaned.lower() in GROUND_ALIASES else cleaned


def _simple_tokens(line: str) -> list[str]:
    normalized = re.sub(r"\s*->\s*", " ", line)
    normalized = normalized.replace(":", " ").replace(",", " ").replace("=", " ")
    return [token for token in normalized.split() if token]


def _parse_component(line: str, *, input_format: str) -> Component:
    tokens = line.split() if input_format == "spice" else _simple_tokens(line)
    if not tokens:
        raise CircuitError("Empty component line.")
    head = tokens[0]
    lower = head.lower()
    if lower in {"wire", "switch", "resistor", "voltage", "current"}:
        if len(tokens) < 2:
            raise CircuitError(f"Missing identifier in line: {line}")
        kind_word, head, tokens = lower, tokens[1], tokens[1:]
    else:
        kind_word = ""
    prefix = head.lower()
    if kind_word == "resistor" or prefix.startswith("r"):
        kind, unit = "resistor", "ohm"
    elif kind_word == "voltage" or prefix.startswith("v"):
        kind, unit = "voltage_source", "V"
    elif kind_word == "current" or prefix.startswith("i"):
        kind, unit = "current_source", "A"
    elif kind_word == "wire" or prefix.startswith("w"):
        kind, unit = "wire", ""
    elif kind_word == "switch" or prefix.startswith("s"):
        kind, unit = "switch", ""
    else:
        kind, unit = f"unsupported:{head[0].upper() if head else 'unknown'}", ""

    if len(tokens) < 3:
        raise CircuitError(f"Expected an identifier and two nodes in line: {line}")
    identifier = tokens[0]
    positive, negative = _node(tokens[1]), _node(tokens[2])
    tail = tokens[3:]
    if tail and tail[0].lower() == "dc":
        tail = tail[1:]
    value: float | None = None
    state: str | None = None
    if kind in {"resistor", "voltage_source", "current_source"}:
        if not tail:
            raise CircuitError(f"Missing value for {identifier}.")
        value = parse_value(tail[0], quantity=f"value for {identifier}")
        if kind == "resistor" and value < 0:
            raise CircuitError(f"Resistance for {identifier} cannot be negative.")
    elif kind == "switch":
        raw_state = tail[0].lower() if tail else ""
        state_aliases = {"on": "closed", "1": "closed", "off": "open", "0": "open"}
        state = state_aliases.get(raw_state, raw_state)
        if state not in {"open", "closed"}:
            raise CircuitError(f"Switch {identifier} requires an explicit OPEN or CLOSED state.")
    extra_count = {"q": 1, "m": 2}.get(prefix[:1], 0) if kind not in SUPPORTED_TYPES else 0
    extra_terminals = [_node(token) for token in tail[:extra_count]]
    return Component(identifier, kind, positive, negative, value, unit, state, line, extra_terminals)


def parse_circuit(circuit_text: str, input_format: Literal["auto", "spice", "simple"] = "auto") -> dict[str, Any]:
    if input_format not in {"auto", "spice", "simple"}:
        raise CircuitError("format must be one of: auto, spice, simple.")
    text = str(circuit_text or "").strip()
    if not text:
        raise CircuitError("Circuit text cannot be empty.")
    components: list[Component] = []
    errors: list[str] = []
    seen: set[str] = set()
    first_words = {line.strip().split()[0].lower().rstrip(":") for line in text.splitlines() if line.strip()}
    detected = (
        "simple"
        if input_format == "auto"
        and (any(mark in text for mark in ("->", ":", ",")) or first_words.intersection({"wire", "switch", "resistor", "voltage", "current"}))
        else input_format
    )
    detected = "spice" if detected == "auto" else detected
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("*", ";", "#", ".")):
            continue
        try:
            component = _parse_component(line, input_format=detected)
            key = component.id.casefold()
            if key in seen:
                raise CircuitError(f"Duplicate component identifier {component.id}.")
            seen.add(key)
            components.append(component)
        except CircuitError as exc:
            errors.append(f"Line {line_number}: {exc}")
    nodes = sorted(
        {node for c in components for node in (c.positive_node, c.negative_node, *c.extra_terminals)},
        key=lambda n: (n != "0", n),
    )
    if "0" not in nodes:
        errors.append("Circuit has no ground reference (use 0, GND, or GROUND).")
    unsupported = [c.id for c in components if c.type not in SUPPORTED_TYPES]
    return {
        "format": detected,
        "components": components,
        "component_graph": [component.to_dict() for component in components],
        "nodes": nodes,
        "errors": errors,
        "unsupported_components": unsupported,
    }


def _collapsed_model(components: list[Component]) -> tuple[_UnionFind, dict[str, str]]:
    nodes = {node for c in components for node in (c.positive_node, c.negative_node)} | {"0"}
    union = _UnionFind(nodes)
    for component in components:
        if component.type == "wire" or (component.type == "switch" and component.state == "closed"):
            union.union(component.positive_node, component.negative_node)
        if component.type == "resistor" and component.value == 0:
            union.union(component.positive_node, component.negative_node)
    ground_root = union.find("0")
    aliases: dict[str, str] = {}
    for node in nodes:
        root = union.find(node)
        aliases[node] = "0" if root == ground_root else root
    return union, aliases


def _connectivity_findings(components: list[Component], aliases: dict[str, str]) -> tuple[list[list[str]], list[str]]:
    graph: dict[str, set[str]] = {node: set() for node in set(aliases.values())}
    for component in components:
        if component.type == "switch" and component.state == "open":
            continue
        a, b = aliases[component.positive_node], aliases[component.negative_node]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    groups: list[list[str]] = []
    unseen = set(graph)
    while unseen:
        start = next(iter(unseen))
        stack, group = [start], set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(graph.get(node, set()) - group)
        unseen -= group
        groups.append(sorted(group))
    floating = sorted(node for group in groups if "0" not in group for node in group)
    return groups, floating


def _solve(components: list[Component]) -> dict[str, Any]:
    _, aliases = _collapsed_model(components)
    groups, floating = _connectivity_findings(components, aliases)
    active = [
        c for c in components
        if c.type in {"resistor", "voltage_source", "current_source"}
        and not (c.type == "resistor" and c.value == 0)
    ]
    nodes = sorted({aliases[n] for c in active for n in (c.positive_node, c.negative_node)} - {"0"})
    node_index = {node: index for index, node in enumerate(nodes)}
    voltage_sources = [c for c in active if c.type == "voltage_source"]
    size = len(nodes) + len(voltage_sources)
    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    def stamp_conductance(a: str, b: str, conductance: float) -> None:
        ia, ib = node_index.get(a), node_index.get(b)
        if ia is not None:
            matrix[ia, ia] += conductance
        if ib is not None:
            matrix[ib, ib] += conductance
        if ia is not None and ib is not None:
            matrix[ia, ib] -= conductance
            matrix[ib, ia] -= conductance

    for component in active:
        a, b = aliases[component.positive_node], aliases[component.negative_node]
        if component.type == "resistor":
            stamp_conductance(a, b, 1.0 / float(component.value))
        elif component.type == "current_source":
            current = float(component.value)
            if a in node_index:
                rhs[node_index[a]] -= current
            if b in node_index:
                rhs[node_index[b]] += current
    for offset, source in enumerate(voltage_sources):
        row = len(nodes) + offset
        a, b = aliases[source.positive_node], aliases[source.negative_node]
        if a in node_index:
            matrix[node_index[a], row] += 1.0
            matrix[row, node_index[a]] += 1.0
        if b in node_index:
            matrix[node_index[b], row] -= 1.0
            matrix[row, node_index[b]] -= 1.0
        rhs[row] = float(source.value)

    if size == 0:
        solution = np.array([], dtype=float)
    else:
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            return {
                "status": "indeterminate",
                "error": "Circuit equations are singular or contradictory; check floating nodes and ideal-source constraints.",
                "detail": str(exc),
                "node_aliases": aliases,
                "connected_groups": groups,
                "floating_nodes": floating,
                "node_voltages": {},
                "branch_currents": {},
                "component_power": {},
            }
    voltages = {"0": 0.0, **{node: float(solution[index]) for node, index in node_index.items()}}
    displayed_voltages = {node: voltages.get(alias, 0.0) for node, alias in aliases.items()}
    currents: dict[str, float] = {}
    power: dict[str, float] = {}
    voltage_source_index = {c.id: len(nodes) + i for i, c in enumerate(voltage_sources)}
    for component in components:
        a, b = aliases[component.positive_node], aliases[component.negative_node]
        voltage_drop = voltages.get(a, 0.0) - voltages.get(b, 0.0)
        current: float | None = None
        if component.type == "resistor" and component.value not in (None, 0):
            current = voltage_drop / float(component.value)
        elif component.type == "current_source":
            current = float(component.value)
        elif component.type == "voltage_source":
            current = float(solution[voltage_source_index[component.id]])
        if current is not None:
            currents[component.id] = current
            power[component.id] = voltage_drop * current
    return {
        "status": "solved",
        "error": "",
        "node_aliases": aliases,
        "connected_groups": groups,
        "floating_nodes": floating,
        "node_voltages": displayed_voltages,
        "branch_currents": currents,
        "component_power": power,
    }


def analyze_circuit(
    circuit_text: str,
    input_format: Literal["auto", "spice", "simple"] = "auto",
    *,
    max_live_dc_voltage: float = 60.0,
) -> dict[str, Any]:
    parsed = parse_circuit(circuit_text, input_format)
    components: list[Component] = parsed.pop("components")
    warnings: list[str] = []
    if parsed["errors"]:
        solution = {"status": "invalid", "error": "Circuit parsing or validation failed."}
    else:
        solution = _solve(components)
    if parsed["unsupported_components"]:
        warnings.append(
            "Unsupported components are included only in the topology; numerical results may not describe paths affected by them."
        )
        if solution.get("status") == "solved":
            solution["status"] = "partial"
    source_voltages = [abs(float(c.value)) for c in components if c.type == "voltage_source" and c.value is not None]
    if source_voltages and max(source_voltages) > float(max_live_dc_voltage):
        warnings.append(
            f"A source exceeds the configured {float(max_live_dc_voltage):g} V DC advisory scope; do not provide energized probing steps."
        )
    floating = solution.get("floating_nodes") or []
    if floating:
        warnings.append(f"Nodes without a ground-connected path: {', '.join(floating)}.")
    return {
        "tool": "analyze_circuit",
        **parsed,
        "summary": {
            "component_count": len(components),
            "node_count": len(parsed["nodes"]),
            "supported_component_count": len(components) - len(parsed["unsupported_components"]),
        },
        "analysis": solution,
        "warnings": warnings,
    }


def _adjacent_components(graph: list[dict[str, Any]], nodes: set[str], component_id: str = "") -> list[str]:
    suspects = []
    for component in graph:
        if component_id and str(component.get("id", "")).casefold() == component_id.casefold():
            suspects.append(str(component["id"]))
        elif nodes.intersection(str(node) for node in component.get("terminals", [])):
            suspects.append(str(component.get("id", "")))
    return sorted({value for value in suspects if value})[:8]


def _equivalent_resistance(graph: list[dict[str, Any]], positive: str, negative: str) -> float | None:
    if any(not component.get("supported", False) for component in graph):
        return None
    nodes = {str(node) for component in graph for node in component.get("terminals", [])} | {"0"}
    if positive not in nodes or negative not in nodes:
        return None
    union = _UnionFind(nodes)
    for component in graph:
        kind = component.get("type")
        value = component.get("value")
        terminals = component.get("terminals", [])
        if len(terminals) != 2:
            continue
        if kind in {"wire", "voltage_source"} or (kind == "switch" and component.get("state") == "closed"):
            union.union(str(terminals[0]), str(terminals[1]))
        if kind == "resistor" and value == 0:
            union.union(str(terminals[0]), str(terminals[1]))
    a, b = union.find(positive), union.find(negative)
    if a == b:
        return 0.0
    collapsed = {node: union.find(node) for node in nodes}
    conductive_nodes = {a, b}
    resistors = []
    for component in graph:
        if component.get("type") != "resistor" or component.get("value") in (None, 0):
            continue
        left, right = (collapsed[str(node)] for node in component["terminals"])
        conductive_nodes.update((left, right))
        resistors.append((left, right, float(component["value"])))
    reference = b
    unknowns = sorted(conductive_nodes - {reference})
    index = {node: i for i, node in enumerate(unknowns)}
    matrix = np.zeros((len(unknowns), len(unknowns)), dtype=float)
    rhs = np.zeros(len(unknowns), dtype=float)
    for left, right, resistance in resistors:
        conductance = 1.0 / resistance
        il, ir = index.get(left), index.get(right)
        if il is not None:
            matrix[il, il] += conductance
        if ir is not None:
            matrix[ir, ir] += conductance
        if il is not None and ir is not None:
            matrix[il, ir] -= conductance
            matrix[ir, il] -= conductance
    rhs[index[a]] = 1.0
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return math.inf
    return max(0.0, float(solution[index[a]]))


def compare_measurements(
    circuit_text: str,
    measurements: list[dict[str, Any]],
    input_format: Literal["auto", "spice", "simple"] = "auto",
    *,
    max_live_dc_voltage: float = 60.0,
    default_tolerance_percent: float = 5.0,
) -> dict[str, Any]:
    circuit = analyze_circuit(circuit_text, input_format, max_live_dc_voltage=max_live_dc_voltage)
    analysis = circuit.get("analysis", {})
    graph = circuit.get("component_graph", [])
    voltages = analysis.get("node_voltages", {}) if analysis.get("status") in {"solved", "partial"} else {}
    currents = analysis.get("branch_currents", {}) if analysis.get("status") in {"solved", "partial"} else {}
    results: list[dict[str, Any]] = []
    hypotheses: dict[str, int] = {}
    for index, raw in enumerate(measurements or [], 1):
        observation = dict(raw) if isinstance(raw, dict) else {}
        kind = str(observation.get("type") or "").strip().lower()
        item: dict[str, Any] = {"index": index, "type": kind, "status": "indeterminate"}
        try:
            if kind == "logic":
                state = str(observation.get("value") or "").strip().lower()
                if state not in {"high", "low", "unknown", "floating"}:
                    raise CircuitError("Logic observations must be HIGH, LOW, UNKNOWN, or FLOATING.")
                item.update({"observed": state, "reason": "No logic threshold was supplied; categorical observation preserved."})
            elif kind == "voltage":
                positive = _node(str(observation.get("positive_node") or observation.get("node") or ""))
                negative = _node(str(observation.get("negative_node") or "0"))
                observed = parse_value(observation.get("value"), quantity="measured voltage")
                if positive not in voltages or negative not in voltages:
                    item.update({"observed": observed, "unit": "V", "reason": "Referenced node is not numerically solved."})
                else:
                    expected = float(voltages[positive]) - float(voltages[negative])
                    item.update(_comparison_fields(observed, expected, observation, default_tolerance_percent, "V"))
                    item["nodes"] = [positive, negative]
            elif kind == "current":
                component_id = str(observation.get("component_id") or "").strip()
                observed = parse_value(observation.get("value"), quantity="measured current")
                matched = next((value for key, value in currents.items() if key.casefold() == component_id.casefold()), None)
                if matched is None:
                    item.update({"observed": observed, "unit": "A", "reason": "Component current is not numerically solved."})
                else:
                    item.update(_comparison_fields(observed, float(matched), observation, default_tolerance_percent, "A"))
                    item["component_id"] = component_id
            elif kind in {"resistance", "continuity"}:
                positive = _node(str(observation.get("positive_node") or observation.get("node") or ""))
                negative = _node(str(observation.get("negative_node") or "0"))
                expected_resistance = _equivalent_resistance(graph, positive, negative)
                item["nodes"] = [positive, negative]
                if expected_resistance is None:
                    item["reason"] = "Equivalent resistance is indeterminate for the supplied topology."
                elif kind == "resistance":
                    observed = parse_value(observation.get("value"), quantity="measured resistance")
                    item.update(
                        _comparison_fields(
                            observed, expected_resistance, observation, default_tolerance_percent, "ohm"
                        )
                    )
                else:
                    raw_value = observation.get("value")
                    if isinstance(raw_value, bool):
                        observed_continuity = raw_value
                    else:
                        normalized = str(raw_value or "").strip().lower()
                        if normalized in {"yes", "true", "closed", "beep", "continuous", "1"}:
                            observed_continuity = True
                        elif normalized in {"no", "false", "open", "no beep", "0"}:
                            observed_continuity = False
                        else:
                            raise CircuitError("Continuity must be yes/no, open/closed, beep/no beep, or boolean.")
                    threshold = parse_value(observation.get("threshold_ohms", 1.0), quantity="continuity threshold")
                    expected_continuity = expected_resistance <= threshold
                    item.update({
                        "status": "match" if observed_continuity == expected_continuity else "mismatch",
                        "observed": observed_continuity,
                        "expected": expected_continuity,
                        "equivalent_resistance_ohms": expected_resistance,
                        "threshold_ohms": threshold,
                    })
            else:
                raise CircuitError("Measurement type must be voltage, current, resistance, continuity, or logic.")
        except CircuitError as exc:
            item["status"] = "invalid"
            item["reason"] = str(exc)
        nodes = {str(item_node) for item_node in item.get("nodes", [])}
        suspects = _adjacent_components(graph, nodes, str(item.get("component_id") or ""))
        item["suspect_components"] = suspects if item.get("status") == "mismatch" else []
        if item.get("status") == "mismatch":
            for suspect in suspects:
                hypotheses[suspect] = hypotheses.get(suspect, 0) + 1
        results.append(item)
    ranked = [
        {"component_id": key, "mismatch_count": count, "classification": "hypothesis_not_confirmed"}
        for key, count in sorted(hypotheses.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return {
        "tool": "compare_measurements",
        "circuit_status": analysis.get("status", "invalid"),
        "measurements": results,
        "ranked_hypotheses": ranked,
        "warnings": circuit.get("warnings", []),
        "errors": circuit.get("errors", []),
        "instructions": "Treat ranked components as hypotheses only; request one discriminating measurement before concluding.",
    }


def _comparison_fields(
    observed: float,
    expected: float,
    observation: dict[str, Any],
    default_tolerance_percent: float,
    unit: str,
) -> dict[str, Any]:
    percent = float(observation.get("tolerance_percent", default_tolerance_percent))
    absolute = float(observation.get("tolerance_absolute", 0.0) or 0.0)
    tolerance = max(abs(expected) * max(0.0, percent) / 100.0, max(0.0, absolute), 1e-12)
    residual = observed - expected
    return {
        "status": "match" if abs(residual) <= tolerance else "mismatch",
        "observed": observed,
        "expected": expected,
        "residual": residual,
        "tolerance": tolerance,
        "unit": unit,
    }
