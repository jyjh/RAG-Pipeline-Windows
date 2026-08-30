"""Shared helpers for browser smoke tests."""
from __future__ import annotations


def activate_tab(page, tab_target: str) -> None:
    page.locator(f'.sidebar-nav .tab[data-tab-target="{tab_target}"]').click()
    page.locator(f"#{tab_target}.panel.active").wait_for(state="visible")
