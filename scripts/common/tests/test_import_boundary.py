"""Non-negotiable: label code must not import reader code (no lens, AR or SAE in labels or admission).

Checks the transitive import closure among scripts/common modules, including imports inside functions (AST walk),
so a label module cannot reach reader code through a shared helper either."""
import ast
from pathlib import Path

COMMON = Path(__file__).resolve().parent.parent
LABEL_MODULES = ["registry", "cells", "wsb", "partners", "margin", "engine", "gates", "stats", "regime", "splits", "brew_material"]
READER_NAMES = {"readers", "jlens", "peft", "vllm", "vllm_lens"}


ENTRY_POINTS = [COMMON.parent / "004_span_assay" / "span_assay.py"]      # experiment scripts that must stay reader-free
MIXED_MODULES = {"leverage_profile", "labels"}                              # 001 / 003 modules (001 holds reader code)


def _imports(name, p=None):
    p = p or COMMON / f"{name}.py"
    if not p.exists():
        return set()
    names = set()
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _closure(name):
    seen, todo = set(), [name]
    while todo:
        m = todo.pop()
        for d in _imports(m):
            if d not in seen:
                seen.add(d)
                if (COMMON / f"{d}.py").exists():
                    todo.append(d)
    return seen


def test_label_modules_do_not_reach_reader_code():
    for m in LABEL_MODULES:
        assert (COMMON / f"{m}.py").exists(), m
        bad = _closure(m) & READER_NAMES
        assert not bad, f"{m}.py reaches reader code: {bad}"


def test_readers_module_is_reader_code():
    assert "jlens" in _closure("readers")    # sanity: the check can see a real reader dependency


def test_reader_free_entry_points():
    """004+ entry points: the import closure (through scripts/common) reaches neither reader code nor the mixed 001 / 003
    modules."""
    for ep in ENTRY_POINTS:
        assert ep.exists(), ep
        direct = _imports(None, ep)
        reach = set(direct)
        for d in direct:
            if (COMMON / f"{d}.py").exists():
                reach |= _closure(d)
        bad = reach & (READER_NAMES | MIXED_MODULES)
        assert not bad, f"{ep.name} reaches {bad}"


def test_entry_point_check_sees_mixed_modules():
    """Sanity: the 003 module imports the mixed 001 module, and the entry-point walk can see it."""
    p = COMMON.parent / "003_labels" / "labels.py"
    assert "leverage_profile" in _imports(None, p)
