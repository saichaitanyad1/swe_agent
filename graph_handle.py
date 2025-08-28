# pip install kuzu google-adk
import uuid
import kuzu
from typing import Dict, Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.callbacks import Callback

# -------------------------
# 0) Process-local registry
# -------------------------
class ObjectRegistry:
    def __init__(self): self._store: Dict[str, Any] = {}
    def put(self, obj) -> str:
        h = str(uuid.uuid4()); self._store[h] = obj; return h
    def get(self, h: str):    return self._store.get(h)
    def pop(self, h: str):    return self._store.pop(h, None)

REGISTRY = ObjectRegistry()

# -------------------------
# 1) Graph context wrapper
# -------------------------
class KuzuGraphCtx:
    """Holds a Kùzu in-memory DB + Connection."""
    def __init__(self):
        # in-memory DB; nothing is persisted after process ends
        db = kuzu.Database(":memory:")  # or "" to mean in-memory
        conn = kuzu.Connection(db)
        self.db = db
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        c = self.conn
        # minimal node/rel schema; tweak to your needs
        c.execute("""CREATE NODE TABLE Node(id STRING PRIMARY KEY, label STRING);""")
        c.execute("""CREATE REL TABLE LINK(FROM Node TO Node, weight INT64);""")

    def add_nodes(self, nodes):
        c = self.conn
        for n in nodes:
            c.execute(f"""CREATE (:{'Node'} {{id: '{n}', label: '{n}'}});""")

    def add_edge(self, u, v, weight=1):
        c = self.conn
        c.execute(f"""
            MATCH (a:Node {{id: '{u}'}}), (b:Node {{id: '{v}'}})
            CREATE (a)-[:LINK {{weight: {int(weight)}}}]->(b);
        """)

    def counts(self):
        c = self.conn
        n = c.execute("MATCH (n:Node) RETURN COUNT(n) AS n;").get_value(0, 0)
        e = c.execute("MATCH ()-[r:LINK]->() RETURN COUNT(r) AS e;").get_value(0, 0)
        return {"nodes": int(n), "edges": int(e)}

# --------------------------------
# 2) Tools that operate by handle
# --------------------------------
def create_graph_tool(seed_nodes: list[str] | None = None, edges: list[tuple] | None = None) -> dict:
    ctx = KuzuGraphCtx()
    if seed_nodes:
        ctx.add_nodes(seed_nodes)
    for u, v, w in (edges or []):
        ctx.add_nodes([u, v])  # idempotent for demo
        ctx.add_edge(u, v, weight=w)
    handle = REGISTRY.put(ctx)
    stats = ctx.counts()
    return {"graph_handle": handle, "stats": stats}

def add_to_graph_tool(graph_handle: str, new_nodes: list[str] | None = None,
                      new_edges: list[tuple] | None = None) -> dict:
    ctx: KuzuGraphCtx | None = REGISTRY.get(graph_handle)
    if not ctx:
        return {"ok": False, "error": f"Unknown handle: {graph_handle}"}
    if new_nodes: 
        ctx.add_nodes(new_nodes)
    for u, v, w in (new_edges or []):
        ctx.add_nodes([u, v])
        ctx.add_edge(u, v, weight=w)
    return {"ok": True, "stats": ctx.counts()}

def query_graph_tool(graph_handle: str, cypher: str) -> dict:
    ctx: KuzuGraphCtx | None = REGISTRY.get(graph_handle)
    if not ctx:
        return {"ok": False, "error": f"Unknown handle: {graph_handle}"}
    qr = ctx.conn.execute(cypher)
    # Return a simple list-of-rows for LLM friendliness
    cols = qr.get_column_names()
    rows = [ [qr.get_value(i, j) for j in range(len(cols))] for i in range(qr.get_num_tuples()) ]
    return {"ok": True, "columns": cols, "rows": rows}

create_graph = FunctionTool(func=create_graph_tool, name="create_graph",
                            description="Create an in-memory Kùzu graph; returns a handle.")
add_to_graph = FunctionTool(func=add_to_graph_tool, name="add_to_graph",
                            description="Add nodes/edges to an existing Kùzu graph by handle.")
query_graph = FunctionTool(func=query_graph_tool, name="query_graph",
                           description="Run a Cypher query on an existing Kùzu graph by handle.")

# ------------------------------------------
# 3) Callback to stash handle into ADK state
# ------------------------------------------
class SaveGraphHandle(Callback):
    async def on_tool_result(self, context, tool_name, result, **_):
        if tool_name == "create_graph" and isinstance(result, dict) and "graph_handle" in result:
            context.state["temp:graph_handle"] = result["graph_handle"]  # available this run across sub-agents

# ------------------------------
# 4) Two-agent handoff pipeline
# ------------------------------
MODEL = "gemini-2.0-flash"

agent_a = LlmAgent(
    name="GraphCreator",
    model=MODEL,
    instruction=(
        "Create an in-memory Kùzu graph with nodes A,B,C and edges A->B (1), B->C (2). "
        "Call create_graph once. Confirm the handle is saved to state."
    ),
    tools=[create_graph],
    output_key="graph_creation_summary",
)

agent_b = LlmAgent(
    name="GraphWorker",
    model=MODEL,
    instruction=(
        "You have a graph handle at {temp:graph_handle}. "
        "Add nodes D,E and edge C->D (5) using add_to_graph. "
        "Then run a Cypher query with query_graph to return all LINKs as (src,dst,weight):\n"
        "MATCH (a:Node)-[r:LINK]->(b:Node) RETURN a.id, b.id, r.weight;"
    ),
    tools=[add_to_graph, query_graph],
    output_key="graph_aug_summary",
)

root = SequentialAgent(name="KuzuGraphPipeline", sub_agents=[agent_a, agent_b])

runner = Runner(
    agent=root,
    app_name="kuzu_handoff_demo",
    session_service=InMemorySessionService(),
    callbacks=[SaveGraphHandle()],
)

# 5) Run (example)
if __name__ == "__main__":
    session = runner.session_service.create_session(app_name="kuzu_handoff_demo", user_id="u1").result()
    events = runner.run_async(user_id="u1", session_id=session.id, new_message="Start")
    for ev in events.result():
        if ev.is_final_response():
            print(ev.stringify_content())
