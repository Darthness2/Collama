"""notebook_edit — get/insert/replace/delete cells in a Jupyter .ipynb."""
from __future__ import annotations

from .base import ToolContext, _resolve, _truncate


def t_notebook_edit(args: dict, ctx: ToolContext) -> str:
    """NotebookEditTool — insert/replace/delete cells in a Jupyter .ipynb file."""
    import json as _json
    path = args["path"]
    op = args.get("op", "replace")  # replace | insert | delete | get
    cell_index = args.get("cell_index")
    new_source = args.get("source", "")
    cell_type = args.get("cell_type", "code")  # code | markdown | raw
    p = _resolve(path, ctx.root)
    if not p.exists():
        if op == "insert" and not cell_index:
            p.parent.mkdir(parents=True, exist_ok=True)
            nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        else:
            return f"ERROR: file not found: {path}"
    else:
        try:
            nb = _json.loads(p.read_text())
        except _json.JSONDecodeError as e:
            return f"ERROR: invalid notebook JSON: {e}"
    cells = nb.setdefault("cells", [])

    if op == "get":
        if cell_index is None:
            return _truncate("\n\n".join(
                f"# cell {i} [{c.get('cell_type','?')}]\n" +
                ("".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))
                for i, c in enumerate(cells)
            ))
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        c = cells[i]
        return f"cell {i} [{c.get('cell_type','?')}]\n" + (
            "".join(c.get("source", [])) if isinstance(c.get("source"), list) else (c.get("source") or ""))

    if not ctx.confirm("notebook edit", f"{op} {path} [cell {cell_index}]"):
        return "ERROR: user denied"

    if op == "replace":
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        cells[i]["source"] = new_source
        if cell_type:
            cells[i]["cell_type"] = cell_type
    elif op == "insert":
        i = len(cells) if cell_index is None else int(cell_index)
        new_cell = {"cell_type": cell_type, "source": new_source, "metadata": {}}
        if cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        cells.insert(max(0, min(i, len(cells))), new_cell)
    elif op == "delete":
        i = int(cell_index)
        if not 0 <= i < len(cells):
            return f"ERROR: cell_index {i} out of range"
        del cells[i]
    else:
        return f"ERROR: unknown op '{op}'"

    p.write_text(_json.dumps(nb, indent=1), encoding="utf-8")
    return f"OK: {op} on {path} (now {len(cells)} cells)"


TOOLS = {"notebook_edit": t_notebook_edit}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "notebook_edit",
        "description": "Get/insert/replace/delete a cell in a Jupyter .ipynb file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "op": {"type": "string", "enum": ["get", "insert", "replace", "delete"]},
            "cell_index": {"type": "integer"},
            "source": {"type": "string"},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"]},
        }, "required": ["path"]},
    }},
]
