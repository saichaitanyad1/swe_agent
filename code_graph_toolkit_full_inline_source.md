Below is the complete, copy‑pasteable source tree for a small Python package that parses repos into a code graph and builds LLM‑ready slices for controller/listener analysis.

---

## 📁 Project tree

```
codegraph-toolkit/
  pyproject.toml
  README.md
  LICENSE
  requirements.txt
  .gitignore
  codegraph/
    __init__.py
    __main__.py
    cli.py
    graph_schema.py
    graph_builder.py
    java_parser.py
    python_parser.py
    exporters.py
    queries.py
    llm_packager.py
    query_engine.py        # dynamic search & natural-language filters
    resolver_java.py       # NEW: import/type-based call resolution for Java
    server.py              # FastAPI server with OpenAPI for Google Agent tools
```

codegraph-toolkit/
pyproject.toml
README.md
LICENSE
requirements.txt
.gitignore
codegraph/
**init**.py
**main**.py
cli.py
graph\_schema.py
graph\_builder.py
java\_parser.py
python\_parser.py
exporters.py
queries.py
llm\_packager.py
query\_engine.py        # NEW: dynamic search & natural-language filters
server.py              # NEW: FastAPI server with OpenAPI for Google Agent tools

````

---

## 🔧 How to use this inline source
1) Create a folder named `codegraph-toolkit` and place the files exactly as below.
2) In a terminal from the project root:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
````

3. Run the CLI:

   ```bash
   codegraph --repo /path/to/repo --lang auto --out ./out --format json graphml mermaid
   codegraph --repo /path/to/repo --slice controllers --neighbors 2 --out ./out --llm-pack controllers
   codegraph --repo /path/to/repo --slice listeners   --neighbors 2 --out ./out --llm-pack listeners
   ```

---

## 📦 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "codegraph-toolkit"
version = "0.2.0"
description = "Parse repos into a rich code graph, expose dynamic search via REST, and build LLM-ready slices."
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "javalang>=0.13.0",
  "networkx>=3.0",
  "fastapi>=0.110",
  "uvicorn>=0.22",
  "pydantic>=2.5"
]

[project.scripts]
codegraph = "codegraph.cli:main"
```

---

## 🧾 `README.md`

````markdown
# CodeGraph Toolkit

Parse a source repo into a code graph (classes, methods, interfaces, calls, overrides, annotations),
export JSON/GraphML/Mermaid, carve LLM-ready slices for **controller**/**listener** analysis, and expose a **REST tool** for agents.

## Install
```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
````

## Quickstart (CLI)

```bash
codegraph --repo /path/to/repo --lang auto --out ./out --format json graphml mermaid
codegraph --repo /path/to/repo --slice controllers --neighbors 2 --out ./out --llm-pack controllers
codegraph --repo /path/to/repo --slice listeners   --neighbors 2 --out ./out --llm-pack listeners
```

## Run as an Agent Tool (REST)

```bash
uvicorn codegraph.server:app --host 0.0.0.0 --port 8080
# Endpoints:
#   POST /build   -> build graph from a repo path (container/local volume)
#   POST /query   -> dynamic search (natural language, regex, annotations, class/method names, k-hops)
#   POST /slice   -> controllers|listeners with neighbors
#   POST /llm-pack-> compact pack + prompt (controllers|listeners)
```

### Google Agent (Agent Builder / Vertex AI) integration

* Deploy the server to Cloud Run with volume or cloud storage access to your repos.
* Register the OpenAPI (FastAPI auto-docs at `/openapi.json`).
* In your agent, call the tool with user NL queries (e.g., "find all Kafka listeners touching OrderService and show callers 2 hops out").

Artifacts:

* `graph.json` / `slice.controllers.json` / `slice.listeners.json`
* `graph.graphml` / `graph.mmd`
* `llm/<scenario>/pack.json` + `prompt.txt`

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Quickstart

```bash
codegraph --repo /path/to/repo --lang auto --out ./out --format json graphml mermaid
codegraph --repo /path/to/repo --slice controllers --neighbors 2 --out ./out --llm-pack controllers
codegraph --repo /path/to/repo --slice listeners   --neighbors 2 --out ./out --llm-pack listeners
# Alternatively, post-process an existing JSON graph:
codegraph --load ./out/graph.json --slice controllers --neighbors 1 --out ./out --llm-pack controllers
```

Artifacts:

* `graph.json` / `slice.controllers.json` / `slice.listeners.json`
* `graph.graphml` (ingest to Neo4j/Gephi/NetworkX)
* `graph.mmd` (Mermaid for quick diagramming)
* `llm/<scenario>/pack.json` + `prompt.txt`

## Notes

* Java parsing uses `javalang`; Python parsing uses builtin `ast`.
* Heuristics can be extended in `codegraph/queries.py`.

````

---

## 📜 `LICENSE`
```text
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

---

## 🧩 `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
*.pyo
*.DS_Store
.out/
/out/
```

---

## 📃 `requirements.txt`

```text
javalang>=0.13.0
networkx>=3.0
fastapi>=0.110
uvicorn>=0.22
pydantic>=2.5
```

---

## 🐍 `codegraph/__init__.py`

```python
from .graph_schema import Node, Edge, NodeType, EdgeType, CodeGraph
```

---

## 🐍 `codegraph/__main__.py`

```python
from .cli import main
if __name__ == "__main__":
    main()
```

---

## 🐍 `codegraph/graph_schema.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Iterable
import json
import networkx as nx

class NodeType(str, Enum):
    FILE = "FILE"
    CLASS = "CLASS"
    INTERFACE = "INTERFACE"
    ENUM = "ENUM"
    METHOD = "METHOD"
    FUNCTION = "FUNCTION"

class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    OVERRIDES = "OVERRIDES"
    CALLS = "CALLS"
    ANNOTATED_BY = "ANNOTATED_BY"
    IMPORTS = "IMPORTS"

@dataclass
class Node:
    id: str
    type: NodeType
    name: str
    fqn: str
    file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    modifiers: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    params: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[str] = None
    doc: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    src: str
    dst: str
    type: EdgeType
    extras: Dict[str, Any] = field(default_factory=dict)

class CodeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.by_fqn = {}

    def add_node(self, node: Node):
        self.g.add_node(node.id, **asdict(node))
        if node.fqn and node.fqn not in self.by_fqn:
            self.by_fqn[node.fqn] = node.id

    def add_edge(self, edge: Edge):
        self.g.add_edge(edge.src, edge.dst, **asdict(edge))

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [self.g.nodes[n] for n in self.g.nodes],
            "edges": [{"src": u, "dst": v, **d} for u, v, d in self.g.edges(data=True)],
        }

    def subgraph_by_nodes(self, node_ids: Iterable[str]) -> "CodeGraph":
        sg = CodeGraph()
        H = self.g.subgraph(node_ids).copy()
        for nid, data in H.nodes(data=True):
            sg.g.add_node(nid, **data)
        for u, v, data in H.edges(data=True):
            sg.g.add_edge(u, v, **data)
        for nid, data in sg.g.nodes(data=True):
            if data.get("fqn"):
                sg.by_fqn[data["fqn"]] = nid
        return sg

    def neighbors_k_hops(self, seeds: Iterable[str], k: int = 1) -> "CodeGraph":
        node_set = set(seeds)
        frontier = set(seeds)
        for _ in range(k):
            new_frontier = set()
            for n in frontier:
                new_frontier.update(self.g.predecessors(n))
                new_frontier.update(self.g.successors(n))
            frontier = new_frontier - node_set
            node_set.update(new_frontier)
        return self.subgraph_by_nodes(node_set)

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    def export_graphml(self, path: str):
        nx.write_graphml(self.g, path)

    def export_mermaid(self, path: str, node_limit: int = 200):
        lines = ["flowchart LR"]
        nodes = list(self.g.nodes)[:node_limit]
        node_set = set(nodes)
        for n in nodes:
            data = self.g.nodes[n]
            label = (data.get("name", n) or "").replace('"', "'")
            lines.append(f'  {n}["{label}\\n({data.get("type")})"]')
        count = 0
        for u, v, d in self.g.edges(data=True):
            if u in node_set and v in node_set:
                et = d.get("type", "EDGE")
                lines.append(f"  {u} -->|{et}| {v}")
                count += 1
                if count > 800:
                    break
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
```

---

## 🐍 `codegraph/java_parser.py`

```python
from __future__ import annotations
from typing import List, Tuple, Dict
import javalang
from .graph_schema import Node, Edge, NodeType, EdgeType


def parse_java_source(src: str, path: str) -> Tuple[List[Node], List[Edge]]:
    """Parse a single Java source file into nodes/edges.
    Captures package, imports, classes/interfaces/enums, fields, methods, annotations,
    extends/implements, and naive calls. Fields are recorded on the class node extras
    for later call resolution (see resolver_java.resolve_calls).
    """
    tree = javalang.parse.parse(src)
    package_name = tree.package.name if tree.package else None
    imports = [imp.path for imp in tree.imports] if tree.imports else []

    nodes: List[Node] = []
    edges: List[Edge] = []

    file_id = f"file::{path}"
    nodes.append(Node(id=file_id, type=NodeType.FILE, name=path.split('/')[-1], fqn=path, file=path))

    def fqn(name: str) -> str:
        return f"{package_name}.{name}" if package_name else name

    def anno_list(annos):
        out = []
        for a in annos or []:
            if hasattr(a, "name"):
                out.append(f"@{a.name}")
        return out

    # Walk type declarations
    for type_decl in tree.types:
        if isinstance(type_decl, javalang.tree.ClassDeclaration):
            ntype = NodeType.CLASS
        elif isinstance(type_decl, javalang.tree.InterfaceDeclaration):
            ntype = NodeType.INTERFACE
        elif isinstance(type_decl, javalang.tree.EnumDeclaration):
            ntype = NodeType.ENUM
        else:
            continue

        class_fqn = fqn(type_decl.name)
        class_id = f"java::{class_fqn}"
        extras: Dict[str, dict] = {"fields": {}}
        class_node = Node(
            id=class_id,
            type=ntype,
            name=type_decl.name,
            fqn=class_fqn,
            file=path,
            line=(type_decl.position.line if type_decl.position else None),
            modifiers=list(type_decl.modifiers or []),
            annotations=anno_list(type_decl.annotations),
            extras=extras,
        )
        nodes.append(class_node)
        edges.append(Edge(src=file_id, dst=class_id, type=EdgeType.CONTAINS))

        # Extends
        if getattr(type_decl, "extends", None):
            exts = type_decl.extends if isinstance(type_decl.extends, list) else [type_decl.extends]
            for e in exts:
                super_fqn = e.name if "." in e.name else fqn(e.name)
                super_id = f"java::{super_fqn}"
                nodes.append(Node(id=super_id, type=NodeType.CLASS, name=super_fqn.split('.')[-1], fqn=super_fqn))
                edges.append(Edge(src=class_id, dst=super_id, type=EdgeType.EXTENDS))

        # Implements
        if getattr(type_decl, "implements", None):
            for i in type_decl.implements or []:
                iface_fqn = i.name if "." in i.name else fqn(i.name)
                iface_id = f"java::{iface_fqn}"
                nodes.append(Node(id=iface_id, type=NodeType.INTERFACE, name=iface_fqn.split('.')[-1], fqn=iface_fqn))
                edges.append(Edge(src=class_id, dst=iface_id, type=EdgeType.IMPLEMENTS))

        # Fields (record types for resolver)
        for body_decl in type_decl.body:
            if isinstance(body_decl, javalang.tree.FieldDeclaration):
                field_type = getattr(body_decl.type, 'name', None)
                for declarator in body_decl.declarators or []:
                    fname = declarator.name
                    class_node.extras.setdefault("fields", {})[fname] = field_type

        # Methods & Calls
        for body_decl in type_decl.body:
            if isinstance(body_decl, javalang.tree.MethodDeclaration):
                method_name = body_decl.name
                params = [{
                    "name": p.name,
                    "type": (p.type.name if p.type else None),
                } for p in (body_decl.parameters or [])]

                return_type = body_decl.return_type.name if getattr(body_decl, "return_type", None) else None
                param_types = ",".join([p["type"] or "var" for p in params])
                method_fqn = f"{class_node.fqn}.{method_name}({param_types})"
                method_id = f"java::{method_fqn}"

                mnode = Node(
                    id=method_id,
                    type=NodeType.METHOD,
                    name=method_name,
                    fqn=method_fqn,
                    file=path,
                    line=(body_decl.position.line if body_decl.position else None),
                    modifiers=list(body_decl.modifiers or []),
                    annotations=anno_list(body_decl.annotations),
                    params=params,
                    returns=return_type,
                )
                nodes.append(mnode)
                edges.append(Edge(src=class_id, dst=method_id, type=EdgeType.CONTAINS))

                # Naive call collection
                if body_decl.body:
                    for _, node2 in body_decl:
                        if isinstance(node2, javalang.tree.MethodInvocation):
                            callee_name = node2.member
                            qualifier = getattr(node2, "qualifier", None)
                            target_fqn_guess = f"{class_node.fqn}.{callee_name}"
                            callee_id = f"java::{target_fqn_guess}"
                            edges.append(Edge(
                                src=method_id,
                                dst=callee_id,
                                type=EdgeType.CALLS,
                                extras={"qualifier": qualifier, "package": package_name, "imports": imports},
                            ))

        # Annotation edges
        for anno in class_node.annotations:
            edges.append(Edge(src=class_id, dst=f"anno::{anno}", type=EdgeType.ANNOTATED_BY))

    # Imports on file node
    for imp in imports:
        edges.append(Edge(src=file_id, dst=f"import::{imp}", type=EdgeType.IMPORTS))

    return nodes, edges
```

---

## 🐍 `codegraph/python_parser.py`

```python
from __future__ import annotations
import ast, os
from typing import List, Tuple
from .graph_schema import Node, Edge, NodeType, EdgeType


def parse_python_source(src: str, path: str) -> Tuple[List[Node], List[Edge]]:
    tree = ast.parse(src)
    nodes: List[Node] = []
    edges: List[Edge] = []

    file_id = f"file::{path}"
    nodes.append(Node(id=file_id, type=NodeType.FILE, name=os.path.basename(path), fqn=path, file=path))

    class Stack:
        def __init__(self):
            self.stack = []
        def push(self, x):
            self.stack.append(x)
        def pop(self):
            return self.stack.pop()
        def top(self):
            return self.stack[-1] if self.stack else None

    st = Stack()

    class Analyzer(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            class_fqn = f"py::{path.replace(os.sep, '/')}.{node.name}"
            cnode = Node(id=class_fqn, type=NodeType.CLASS, name=node.name, fqn=class_fqn, file=path, line=node.lineno)
            nodes.append(cnode)
            edges.append(Edge(src=file_id, dst=class_fqn, type=EdgeType.CONTAINS))
            for b in node.bases:
                if isinstance(b, ast.Name):
                    edges.append(Edge(src=class_fqn, dst=f"py::sym::{b.id}", type=EdgeType.EXTENDS))
                elif isinstance(b, ast.Attribute):
                    edges.append(Edge(src=class_fqn, dst=f"py::sym::{b.attr}", type=EdgeType.EXTENDS))
            st.push(class_fqn)
            self.generic_visit(node)
            st.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            parent = st.top()
            params = [{"name": a.arg, "type": None} for a in node.args.args]
            fn_id = f"py::{path.replace(os.sep, '/')}.{node.name}({','.join([p['name'] for p in params])})"
            ntype = NodeType.METHOD if parent else NodeType.FUNCTION
            fnode = Node(id=fn_id, type=ntype, name=node.name, fqn=fn_id, file=path, line=node.lineno, params=params)
            nodes.append(fnode)
            if parent:
                edges.append(Edge(src=parent, dst=fn_id, type=EdgeType.CONTAINS))
            else:
                edges.append(Edge(src=file_id, dst=fn_id, type=EdgeType.CONTAINS))
            for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                if isinstance(call.func, ast.Name):
                    callee = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    callee = call.func.attr
                else:
                    callee = "call"
                callee_id = f"py::sym::{callee}"
                edges.append(Edge(src=fn_id, dst=callee_id, type=EdgeType.CALLS))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)

    Analyzer().visit(tree)
    return nodes, edges
```

---

## 🐍 `codegraph/graph_builder.py`

```python
from __future__ import annotations
import os, io
from .graph_schema import CodeGraph, Node, Edge, NodeType, EdgeType
from .java_parser import parse_java_source
from .python_parser import parse_python_source
from .resolver_java import resolve_calls as resolve_calls_java


def build_graph_from_repo(repo_path: str, lang: str = "auto") -> CodeGraph:
    G = CodeGraph()
    for root, _, files in os.walk(repo_path):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            if lang == "java" or (lang == "auto" and ext == ".java"):
                with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                    src = f.read()
                try:
                    nodes, edges = parse_java_source(src, path)
                    for n in nodes:
                        G.add_node(n)
                    for e in edges:
                        G.add_edge(e)
                except Exception as e:
                    file_id = f"file::{path}"
                    G.add_node(Node(id=file_id, type=NodeType.FILE, name=fn, fqn=path, file=path, extras={"parse_error": str(e)}))
            elif lang == "python" or (lang == "auto" and ext == ".py"):
                with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                    src = f.read()
                try:
                    nodes, edges = parse_python_source(src, path)
                    for n in nodes:
                        G.add_node(n)
                    for e in edges:
                        G.add_edge(e)
                except Exception as e:
                    file_id = f"file::{path}"
                    G.add_node(Node(id=file_id, type=NodeType.FILE, name=fn, fqn=path, file=path, extras={"parse_error": str(e)}))
    derive_overrides(G)
    # NEW: resolve Java CALLS edges using imports/package/field types
    resolve_calls_java(G)
    return G


def derive_overrides(G: CodeGraph):
    """Simple override inference: if class A extends B, and A.m(..) matches B.m(..) by name+arity, mark OVERRIDES."""
    supers = {}
    for u, v, d in G.g.edges(data=True):
        if d.get("type") == EdgeType.EXTENDS:
            supers.setdefault(u, set()).add(v)

    class_methods = {}
    for nid, data in G.g.nodes(data=True):
        if data.get("type") == NodeType.METHOD:
            fqn = data.get("fqn") or ""
            cls = fqn.rsplit(".", 1)[0] if "." in fqn else None
            name = data.get("name")
            arity = len(data.get("params", []))
            if cls:
                class_methods.setdefault(cls, []).append((nid, name, arity))

    for cls, meths in class_methods.items():
        cls_node_id = G.by_fqn.get(cls) or f"java::{cls}"
        queue = list(supers.get(cls_node_id, []))
        seen = set(queue)
        hops = 0
        while queue and hops < 3:
            nxt = []
            for n in queue:
                for _, sup, d in G.g.out_edges(n, data=True):
                    if d.get("type") == EdgeType.EXTENDS and sup not in seen:
                        seen.add(sup)
                        nxt.append(sup)
            queue = nxt
            hops += 1

        super_meths = []
        for sn in seen:
            for _, mn, d in G.g.out_edges(sn, data=True):
                if d.get("type") == EdgeType.CONTAINS:
                    mdata = G.g.nodes[mn]
                    if mdata.get("type") == NodeType.METHOD:
                        super_meths.append((mn, mdata.get("name"), len(mdata.get("params", []))))

        super_index = {}
        for mn, name, ar in super_meths:
            super_index.setdefault((name, ar), []).append(mn)

        for nid, name, ar in meths:
            for super_n in super_index.get((name, ar), []):
                G.add_edge(Edge(src=nid, dst=super_n, type=EdgeType.OVERRIDES))
```

---

## 🐍 `codegraph/exporters.py`

```python
from __future__ import annotations
from typing import Iterable
from .graph_schema import CodeGraph, NodeType, EdgeType


def export_json(G: CodeGraph, path: str):
    G.export_json(path)


def export_graphml(G: CodeGraph, path: str):
    G.export_graphml(path)


def export_mermaid(G: CodeGraph, path: str, node_limit: int = 200):
    G.export_mermaid(path, node_limit=node_limit)


def compact_for_llm(
    G: CodeGraph,
    token_budget_nodes: int = 400,
    include_edges: Iterable[EdgeType] = (
        EdgeType.CONTAINS,
        EdgeType.CALLS,
        EdgeType.EXTENDS,
        EdgeType.IMPLEMENTS,
        EdgeType.OVERRIDES,
    ),
):
    # Keep top-N nodes by priority then degree
    priority = {
        NodeType.CLASS: 3,
        NodeType.INTERFACE: 3,
        NodeType.ENUM: 2,
        NodeType.METHOD: 2,
        NodeType.FUNCTION: 2,
        NodeType.FILE: 1,
    }
    nodes = list(G.g.nodes())
    scored = []
    for n in nodes:
        d = G.g.nodes[n]
        deg = G.g.degree[n]
        scored.append((priority.get(d.get("type"), 0), deg, n))
    scored.sort(reverse=True)
    keep = set(n for _, _, n in scored[:token_budget_nodes])
    H = G.subgraph_by_nodes(keep)

    # strip bulky attrs
    for n in list(H.g.nodes()):
        if "doc" in H.g.nodes[n]:
            H.g.nodes[n]["doc"] = None
        if "extras" in H.g.nodes[n]:
            H.g.nodes[n]["extras"] = None

    # filter edges by type
    to_drop = []
    for u, v, e in H.g.edges(keys=True):
        et = H.g.edges[u, v, e].get("type")
        if et not in include_edges:
            to_drop.append((u, v, e))
    for u, v, e in to_drop:
        H.g.remove_edge(u, v, key=e)
    return H
```

---

## 🐍 `codegraph/queries.py`

```python
from __future__ import annotations
from typing import Set
from .graph_schema import CodeGraph, NodeType, EdgeType

CONTROLLER_ANNOS = {"@Controller", "@RestController"}
REQUEST_ANNOS = {"@RequestMapping", "@GetMapping", "@PostMapping", "@PutMapping", "@PatchMapping", "@DeleteMapping"}
LISTENER_ANNOS = {"@EventListener", "@KafkaListener", "@RabbitListener", "@JmsListener"}
LISTENER_INTERFACES = {"ApplicationListener", "MessageListener"}


def slice_controllers(G: CodeGraph, neighbors: int = 1) -> CodeGraph:
    seeds: Set[str] = set()
    for n, d in G.g.nodes(data=True):
        if d.get("type") == NodeType.CLASS and any(a in CONTROLLER_ANNOS for a in d.get("annotations", [])):
            seeds.add(n)
    for n, d in G.g.nodes(data=True):
        if d.get("type") == NodeType.METHOD and any(a in REQUEST_ANNOS for a in d.get("annotations", [])):
            seeds.add(n)
    if not seeds:
        return G.subgraph_by_nodes(set())
    return G.neighbors_k_hops(seeds, k=neighbors)


def slice_listeners(G: CodeGraph, neighbors: int = 1) -> CodeGraph:
    seeds: Set[str] = set()
    for n, d in G.g.nodes(data=True):
        if d.get("type") == NodeType.CLASS and any(a in LISTENER_ANNOS for a in d.get("annotations", [])):
            seeds.add(n)
    for u, v, data in G.g.edges(data=True):
        if data.get("type") == EdgeType.IMPLEMENTS:
            iface = G.g.nodes.get(v, {}).get("name") or ""
            if any(iface.endswith(LI) for LI in LISTENER_INTERFACES) or any(str(v).endswith(LI) for LI in LISTENER_INTERFACES):
                seeds.add(u)
    for n, d in G.g.nodes(data=True):
        if d.get("type") == NodeType.METHOD and any(a in LISTENER_ANNOS for a in d.get("annotations", [])):
            seeds.add(n)
    if not seeds:
        return G.subgraph_by_nodes(set())
    return G.neighbors_k_hops(seeds, k=neighbors)
```

---

## 🐍 `codegraph/llm_packager.py`

```python
from __future__ import annotations
from typing import Literal
from .exporters import compact_for_llm

PROMPT_CONTROLLERS = (
    "You are a senior backend reviewer. Analyze the provided code graph focused on web controllers.
"
    "Goal: identify endpoints, auth boundaries, cross-service calls, and risky patterns.
"
    "Return a structured report with:
"
    "1) Endpoint inventory (HTTP method, path if available, controller class, method name).
"
    "2) Downstream calls per endpoint (internal services, DAOs, http clients).
"
    "3) Security notes (missing auth/validation, deserialization of request bodies, exception leakage).
"
    "4) Duplications or dead endpoints (no callers, overlapping paths).
"
    "5) Suggestions: refactoring and tests.
"
    "Only use the provided graph. If data is missing, call it out explicitly.
"
)

PROMPT_LISTENERS = (
    "You are a senior backend reviewer. Analyze the provided code graph focused on event/message listeners.
"
    "Return a structured report with:
"
    "1) Listener inventory (annotation or interface, event/topic/queue if available, class.method).
"
    "2) Fan-in / fan-out: what triggers these listeners and what they call downstream.
"
    "3) Ordering, retries, idempotency signals; potential dead-letter handling or lack thereof.
"
    "4) Concurrency hotspots or long-running work in listeners.
"
    "5) Suggestions: backpressure safeguards, poison message handling, observability.
"
    "Only use the provided graph. If data is missing, call it out explicitly.
"
)


def write_llm_pack(G, out_dir: str, prompt_text: str):
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pack.json"), "w", encoding="utf-8") as f:
        json.dump(G.to_json(), f)
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)


def build_llm_pack(graph, scenario: Literal["controllers", "listeners"], out_dir: str):
    if scenario == "controllers":
        pack = compact_for_llm(graph, token_budget_nodes=400)
        write_llm_pack(pack, out_dir, PROMPT_CONTROLLERS)
    elif scenario == "listeners":
        pack = compact_for_llm(graph, token_budget_nodes=400)
        write_llm_pack(pack, out_dir, PROMPT_LISTENERS)
    else:
        raise ValueError("unknown scenario")
```

---

## 🐍 `codegraph/cli.py`

```python
from __future__ import annotations
import argparse, os, json
from .graph_builder import build_graph_from_repo
from .exporters import export_json, export_graphml, export_mermaid
from .queries import slice_controllers, slice_listeners
from .llm_packager import build_llm_pack


def main():
    ap = argparse.ArgumentParser(description="CodeGraph Toolkit CLI")
    ap.add_argument("--repo", help="path to source repo")
    ap.add_argument("--lang", default="auto", choices=["auto", "java", "python"], help="language mode")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--format", nargs="*", default=["json"], choices=["json", "graphml", "mermaid"], help="export formats")
    ap.add_argument("--slice", choices=["controllers", "listeners", "none"], default="none", help="optional slice")
    ap.add_argument("--neighbors", type=int, default=1, help="k-hop neighbors for slice")
    ap.add_argument("--llm-pack", choices=["controllers", "listeners", "none"], default="none", help="build compact LLM pack for scenario")
    ap.add_argument("--load", help="load an existing graph.json instead of parsing repo")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.load:
        with open(args.load, "r", encoding="utf-8") as f:
            data = json.load(f)
        from .graph_schema import CodeGraph
        G = CodeGraph()
        for n in data["nodes"]:
            G.g.add_node(n["id"], **n)
            if n.get("fqn"):
                G.by_fqn[n["fqn"]] = n["id"]
        for e in data["edges"]:
            src, dst = e.pop("src"), e.pop("dst")
            G.g.add_edge(src, dst, **e)
    else:
        if not args.repo:
            ap.error("--repo is required when not using --load")
        G = build_graph_from_repo(args.repo, lang=args.lang)

    graph_for_export = G

    if args.slice == "controllers":
        graph_for_export = slice_controllers(G, neighbors=args.neighbors)
    elif args.slice == "listeners":
        graph_for_export = slice_listeners(G, neighbors=args.neighbors)

    base = os.path.join(args.out, "graph" if args.slice == "none" else f"slice.{args.slice}")
    if "json" in args.format:
        export_json(graph_for_export, base + ".json")
    if "graphml" in args.format:
        export_graphml(graph_for_export, base + ".graphml")
    if "mermaid" in args.format:
        export_mermaid(graph_for_export, base + ".mmd")

    if args.llm_pack in ("controllers", "listeners"):
        out_dir = os.path.join(args.out, "llm", args.llm_pack)
        build_llm_pack(graph_for_export, args.llm_pack, out_dir)


if __name__ == "__main__":
    main()
```

---

## ✅ Sanity test (optional)

After installing, create a tiny sample project with one Java controller and a Python helper to verify graph construction. Then run `codegraph` as shown above and inspect `out/graph.json` and `out/graph.mmd`.

---

## 🐍 `codegraph/resolver_java.py`

```python
from __future__ import annotations
from typing import List, Dict, Optional
import re
from .graph_schema import CodeGraph, EdgeType, NodeType, Edge

# Heuristic Java call resolver:
# - Uses CALLS edge extras: qualifier, package, imports
# - Uses class extras.fields to map field name -> declared type
# - Resolves targets to existing class METHOD nodes by name (any arity match)


def resolve_calls(G: CodeGraph):
    # Build quick indexes
    class_nodes_by_simple: Dict[str, List[str]] = {}
    for nid, data in G.g.nodes(data=True):
        if data.get("type") == NodeType.CLASS:
            simple = (data.get("name") or "").split(".")[-1]
            class_nodes_by_simple.setdefault(simple, []).append(nid)

    def find_class_fqn(simple: str, pkg: Optional[str], imports: List[str]) -> Optional[str]:
        # 1) Exact import match
        for imp in imports or []:
            if imp.endswith("." + simple) or imp == simple:
                return imp
        # 2) Same package guess
        if pkg:
            guess = f"{pkg}.{simple}"
            # Validate exists in graph
            nid = G.by_fqn.get(guess)
            if nid:
                return guess
        # 3) Any class with this simple name (fallback)
        for nid in class_nodes_by_simple.get(simple, []):
            return G.g.nodes[nid].get("fqn")
        return None

    # Collect CALLS edges to potentially rewrite
    calls = []
    for u, v, k, d in G.g.edges(keys=True, data=True):
        if d.get("type") == EdgeType.CALLS:
            calls.append((u, v, k, d))

    for u, v, k, d in calls:
        qual = (d.get("extras") or {}).get("qualifier")
        pkg = (d.get("extras") or {}).get("package")
        imports = (d.get("extras") or {}).get("imports") or []
        callee_guess = G.g.nodes[v].get("name") or G.g.nodes[v].get("fqn") or ""
        # derive member name from callee node fqn or edge context
        member = None
        target_name = G.g.nodes[v].get("name")
        # name is method simple name for METHOD nodes, but for our placeholder it's the method name too
        if target_name:
            member = target_name
        # Try qualifier-based resolution
        resolved_class_fqn = None
        if qual:
            # field? check class of caller
            # Find caller class via CONTAINS edge (class -> method = u)
            caller_class = None
            for pred, _, ed in G.g.in_edges(u, data=True):
                if ed.get("type") == EdgeType.CONTAINS and G.g.nodes[pred].get("type") == NodeType.CLASS:
                    caller_class = pred
                    break
            if caller_class:
                fields = (G.g.nodes[caller_class].get("extras") or {}).get("fields", {})
                ftype = fields.get(qual)
                if ftype:
                    resolved_class_fqn = find_class_fqn(ftype, pkg, imports)
            # Maybe qualifier is a Class name directly
            if not resolved_class_fqn and qual[:1].upper() == qual[:1]:
                resolved_class_fqn = find_class_fqn(qual, pkg, imports)
        else:
            # unqualified -> same class
            caller_class = None
            for pred, _, ed in G.g.in_edges(u, data=True):
                if ed.get("type") == EdgeType.CONTAINS and G.g.nodes[pred].get("type") == NodeType.CLASS:
                    caller_class = pred
                    break
            if caller_class:
                resolved_class_fqn = G.g.nodes[caller_class].get("fqn")

        if not (resolved_class_fqn and member):
            continue

        # find candidate method nodes under resolved class by name
        target_class_node = G.by_fqn.get(resolved_class_fqn)
        if not target_class_node:
            continue
        method_targets = []
        for _, mn, ed in G.g.out_edges(target_class_node, data=True):
            if ed.get("type") == EdgeType.CONTAINS and G.g.nodes[mn].get("type") == NodeType.METHOD:
                if G.g.nodes[mn].get("name") == member:
                    method_targets.append(mn)

        if not method_targets:
            continue

        # Rewire: remove old edge and add resolved edges
        G.g.remove_edge(u, v, key=k)
        for t in method_targets:
            G.add_edge(Edge(src=u, dst=t, type=EdgeType.CALLS, extras={"resolved": True}))
```

## 🐍 `codegraph/query_engine.py`

```python
from __future__ import annotations
from typing import Dict, Any, Optional
import re
from .graph_schema import CodeGraph, NodeType, EdgeType

# Lightweight dynamic query engine for natural-language-like filters.
# Filters (all optional): text, kind, annotations_any, name_regex, file_regex,
# calls, implements, extends, neighbors.

KIND_MAP = {
    "file": NodeType.FILE,
    "class": NodeType.CLASS,
    "interface": NodeType.INTERFACE,
    "enum": NodeType.ENUM,
    "method": NodeType.METHOD,
    "function": NodeType.FUNCTION,
}

def _match_text(s: Optional[str], needle: str) -> bool:
    return bool(s and needle.lower() in s.lower())

def dynamic_query(G: CodeGraph, q: Dict[str, Any]) -> CodeGraph:
    text = q.get("text")
    kind = q.get("kind")
    annos = set(q.get("annotations_any") or [])
    name_regex = q.get("name_regex")
    file_regex = q.get("file_regex")
    calls = q.get("calls")
    implements = q.get("implements")
    extends = q.get("extends")
    neighbors = int(q.get("neighbors") or 0)

    kind_enum = KIND_MAP.get(kind.lower()) if isinstance(kind, str) else None
    name_rx = re.compile(name_regex) if name_regex else None
    file_rx = re.compile(file_regex) if file_regex else None

    seeds = []
    for n, d in G.g.nodes(data=True):
        if kind_enum and d.get("type") != kind_enum:
            continue
        if text and not (
            _match_text(d.get("name"), text)
            or _match_text(d.get("fqn"), text)
            or any(_match_text(a, text) for a in d.get("annotations", []))
            or _match_text(d.get("file"), text)
        ):
            continue
        if annos and not (set(d.get("annotations", [])) & annos):
            continue
        if name_rx and not (name_rx.search(d.get("name") or "") or name_rx.search(d.get("fqn") or "")):
            continue
        if file_rx and not file_rx.search(d.get("file") or ""):
            continue
        seeds.append(n)

    if implements:
        seeds = [n for n in seeds if any(
            data.get("type") == EdgeType.IMPLEMENTS and (str(v).endswith(implements) or (G.g.nodes.get(v,{}).get("name"," ").endswith(implements)))
            for _, v, data in G.g.out_edges(n, data=True)
        )]
    if extends:
        seeds = [n for n in seeds if any(
            data.get("type") == EdgeType.EXTENDS and (str(v).endswith(extends) or (G.g.nodes.get(v,{}).get("name"," ").endswith(extends)))
            for _, v, data in G.g.out_edges(n, data=True)
        )]
    if calls:
        seeds = [n for n in seeds if any(
            data.get("type") == EdgeType.CALLS and calls.lower() in (G.g.nodes.get(v,{}).get("fqn"," ") + G.g.nodes.get(v,{}).get("name"," ")).lower()
            for _, v, data in G.g.out_edges(n, data=True)
        )]

    if not seeds:
        return G.subgraph_by_nodes(set())
    return G.neighbors_k_hops(seeds, k=neighbors) if neighbors > 0 else G.subgraph_by_nodes(seeds)
```

---

## 🐍 `codegraph/server.py`

```python
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os

from .graph_builder import build_graph_from_repo
from .graph_schema import CodeGraph
from .queries import slice_controllers, slice_listeners
from .llm_packager import build_llm_pack
from .query_engine import dynamic_query

app = FastAPI(title="CodeGraph Toolkit API", version="0.2.0")

GRAPH: Optional[CodeGraph] = None

class BuildReq(BaseModel):
    repo: str = Field(..., description="Path to source repo mounted in container")
    lang: str = Field("auto", description="auto|java|python")

class QueryReq(BaseModel):
    query: Dict[str, Any] = Field(default_factory=dict)

class SliceReq(BaseModel):
    kind: str = Field(..., description="controllers|listeners")
    neighbors: int = 1

class PackReq(BaseModel):
    scenario: str = Field(..., description="controllers|listeners")
    out_dir: str = "./out/llm"

@app.post("/build")
def build(req: BuildReq):
    global GRAPH
    GRAPH = build_graph_from_repo(req.repo, lang=req.lang)
    return {"status": "ok", "nodes": GRAPH.g.number_of_nodes(), "edges": GRAPH.g.number_of_edges()}

@app.post("/query")
def query(req: QueryReq):
    if GRAPH is None:
        return {"error": "graph not built; call /build first"}
    sg = dynamic_query(GRAPH, req.query)
    return sg.to_json()

@app.post("/slice")
def slice(req: SliceReq):
    if GRAPH is None:
        return {"error": "graph not built; call /build first"}
    if req.kind == "controllers":
        sg = slice_controllers(GRAPH, neighbors=req.neighbors)
    elif req.kind == "listeners":
        sg = slice_listeners(GRAPH, neighbors=req.neighbors)
    else:
        return {"error": "unknown slice kind"}
    return sg.to_json()

@app.post("/llm-pack")
def pack(req: PackReq):
    if GRAPH is None:
        return {"error": "graph not built; call /build first"}
    out = os.path.join(req.out_dir, req.scenario)
    build_llm_pack(GRAPH, req.scenario, out)
    return {"status": "ok", "out": out}
```
