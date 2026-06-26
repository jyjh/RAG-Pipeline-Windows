from __future__ import annotations


class QueryEngine:
    def __new__(cls, *args, **kwargs):
        from src.query import QueryEngine as _QueryEngine

        return _QueryEngine(*args, **kwargs)
