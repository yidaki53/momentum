"""Static smoke tests for mobile button wiring in KV.

These checks run without importing Kivy, so they can guard CI environments
where graphical dependencies are unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "mobile" / "main.py").read_text(encoding="utf-8")


def _kv_text(src: str) -> str:
    match = re.search(r'KV\s*=\s*"""(.*?)"""', src, re.DOTALL)
    assert match, "KV string block not found in mobile/main.py"
    return match.group(1)


def _class_body(src: str, class_name: str) -> str:
    pattern = rf"^class {class_name}\b.*?:\n(.*?)(?=^class |\Z)"
    match = re.search(pattern, src, re.DOTALL | re.MULTILINE)
    assert match, f"class {class_name} not found"
    return match.group(1)


def _methods_from_class(src: str, class_name: str) -> set[str]:
    body = _class_body(src, class_name)
    return set(re.findall(r"^\s+def\s+([a-zA-Z_]\w*)\(", body, re.MULTILINE))


def _kv_rule_blocks(kv: str) -> dict[str, str]:
    rules: dict[str, str] = {}
    for match in re.finditer(
        r"^<([A-Za-z_][\w@]*)>:\n(.*?)(?=^<[A-Za-z_][\w@]*>:\n|\Z)",
        kv,
        re.DOTALL | re.MULTILINE,
    ):
        rules[match.group(1)] = match.group(2)
    return rules


def _screen_names_from_build(src: str) -> set[str]:
    return set(re.findall(r'sm\.add_widget\(\w+\(name="([^"]+)"\)\)', src))


def test_root_handlers_exist_for_kv_rules() -> None:
    src = _source()
    kv = _kv_text(src)
    rules = _kv_rule_blocks(kv)

    rule_to_class = {
        "HomeScreen": "HomeScreen",
        "Toolbar": "Toolbar",
        "HelpMenuScreen": "HelpMenuScreen",
        "TestsMenuScreen": "TestsMenuScreen",
        "HowToScreen": "HowToScreen",
        "ScienceScreen": "ScienceScreen",
        "AboutScreen": "AboutScreen",
        "StroopScreen": "StroopScreen",
    }

    for rule_name, class_name in rule_to_class.items():
        if rule_name not in rules:
            continue
        methods = _methods_from_class(src, class_name)
        root_calls = set(
            re.findall(r"on_release:\s*root\.([a-zA-Z_]\w*)\(", rules[rule_name])
        )
        missing = sorted(name for name in root_calls if name not in methods)
        assert not missing, (
            f"{rule_name} has on_release handlers missing in {class_name}: {missing}"
        )


def test_app_handlers_exist_for_kv_rules() -> None:
    src = _source()
    kv = _kv_text(src)
    app_methods = _methods_from_class(src, "MomentumApp")
    app_calls = set(re.findall(r"on_release:\s*app\.([a-zA-Z_]\w*)\(", kv))
    missing = sorted(name for name in app_calls if name not in app_methods)
    assert not missing, f"KV app handlers missing in MomentumApp: {missing}"


def test_toolbar_screen_targets_exist_in_screen_manager() -> None:
    src = _source()
    kv = _kv_text(src)
    screen_names = _screen_names_from_build(src)
    toolbar_rule = _kv_rule_blocks(kv).get("Toolbar", "")

    # on_release: root.go('home', 'right')
    targets = set(
        re.findall(r"on_release:\s*root\.go\('([^']+)',\s*'[^']+'\)", toolbar_rule)
    )
    missing = sorted(name for name in targets if name not in screen_names)
    assert not missing, f"Toolbar has unknown screen targets: {missing}"
