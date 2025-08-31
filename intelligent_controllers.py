import os, json, glob, csv, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set

import javalang
import networkx as nx
import yaml

# ---- Google ADK imports ----
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field, conint

# ================= FS TOOL (deterministic) =================
class FsTool:
    def list_files(self, root: str, patterns: List[str]) -> List[str]:
        out = []
        for pat in patterns:
            out.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
        return [p for p in out if os.path.isfile(p)]

    def read_head(self, path: str, max_bytes: int = 32_000) -> str:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="ignore")

    def sample_java_files(self, root: str, max_files: int = 40) -> List[Dict[str,str]]:
        all_java = self.list_files(root, ["*.java"])
        # prioritize likely web modules by simple heuristics
        prio = [p for p in all_java if any(x in p.lower() for x in ["controller", "web", "rest", "api"])]
        prio = prio[:max_files]
        if len(prio) < max_files:
            extra = [p for p in all_java if p not in prio][:max_files-len(prio)]
            prio.extend(extra)
        return [{"path": p, "head": self.read_head(p)} for p in prio]

    def sample_yml(self, root: str, max_files: int = 6) -> List[Dict[str,str]]:
        ymls = self.list_files(root, ["application.yml", "application.yaml",
                                      "application-*.yml", "application-*.yaml"])[:max_files]
        out=[]
        for p in ymls:
            try:
                out.append({"path": p, "text": self.read_head(p, 200_000)})
            except Exception:
                pass
        return out

# ============= LLM Controller Suggester ====================
class SuggesterInput(BaseModel):
    repo_root: str
    max_depth: conint(ge=1, le=6) = 2
    include_pkg_prefix: Optional[str] = None

class SuggesterOutput(BaseModel):
    # Suggested controller handlers (LLM guessed)
    suggestions: List[Dict[str, Any]] = Field(
        description="Each: {file, fqcn?, class_name, method, http_method?, paths?, rationale}"
    )

SYSTEM_PROMPT = """\
You are an expert code analyst for Java/Spring and custom Spring-like frameworks.
Given snippets of .java files and application.yml, find probable HTTP controller methods.
Be flexible: teams may use composed annotations (e.g., @ApiController), base classes, or DSLs.

Return a JSON with a 'suggestions' array. Each item:
- file: absolute or repo-relative path to the file
- class_name: the class containing the handler
- fqcn (best effort): package.Class
- method: method name
- http_method (best effort): GET|POST|PUT|DELETE|PATCH|ANY
- paths (best effort): array of strings
- rationale: 1-2 sentences explaining your inference

Rules:
- Prefer high precision over recall; only include likely handlers.
- If annotation names are unknown but clearly map to HTTP routes, include them with http_method='ANY'.
- Do NOT invent files or classes not shown in snippets.
"""

controller_suggester = LlmAgent(
    name="ControllerSuggester",
    model="gemini-2.0-flash",
    include_contents="none",
    instruction=SYSTEM_PROMPT,
    input_schema=SuggesterInput,
    output_schema=SuggesterOutput,
    output_key="controller_suggestions_json"
)

# ============= Deterministic Verifier (AST) =================
HTTP_SIMPLE = {"RequestMapping":"ANY","GetMapping":"GET","PostMapping":"POST","PutMapping":"PUT",
               "DeleteMapping":"DELETE","PatchMapping":"PATCH"}

def _ann_name(a) -> str:
    nm = getattr(a, "name", "")
    return nm.split(".")[-1] if nm else ""

def _ann_vals(a) -> Dict[str, Any]:
    vals={}
    el=getattr(a,"element",None)
    if isinstance(el, list):
        for pair in el:
            vals[pair.name]=getattr(pair.value,"value",getattr(pair,"value",None))
    elif el is not None and not hasattr(el,"name"):
        vals["value"]=getattr(el,"value",el)
    return vals

def _paths_from_vals(vals: Dict[str,Any]) -> List[str]:
    out=[]
    for k in ("path","value"):
        v=vals.get(k)
        if v is None: continue
        if isinstance(v,(list,tuple)):
            out.extend([str(x) for x in v])
        else:
            out.append(str(v))
    return out or [""]

@dataclass
class VerifiedHandler:
    fqmn: str
    http_method: str
    paths: List[str]
    file: str

def verify_and_normalize(repo_root: str, suggestions: List[Dict[str,Any]]) -> List[VerifiedHandler]:
    """Open suggested files, parse with javalang, confirm handlers, normalize fqmn/verb/paths."""
    verified: List[VerifiedHandler] = []

    # Group suggestions by file to avoid reparsing
    by_file: Dict[str, List[Dict[str,Any]]] = {}
    for s in suggestions:
        fp = s["file"]
        if not os.path.isabs(fp):
            fp = os.path.abspath(os.path.join(repo_root, fp))
        by_file.setdefault(fp, []).append({**s, "file": fp})

    for file_path, group in by_file.items():
        if not os.path.exists(file_path):
            continue
        try:
            tree = javalang.parse.parse(open(file_path, "r", encoding="utf-8", errors="ignore").read())
        except Exception:
            continue

        pkg = getattr(tree.package,"name",None)
        for t in getattr(tree,"types",[]) or []:
            if not hasattr(t,"name"): continue
            class_name=t.name
            fqcn = f"{pkg+'.' if pkg else ''}{class_name}"
            class_annos = list(getattr(t,"annotations",[]) or [])
            class_map_paths=[]
            for a in class_annos:
                nm=_ann_name(a)
                if nm in HTTP_SIMPLE:
                    class_map_paths = _paths_from_vals(_ann_vals(a))
                    break

            for m in getattr(t,"methods",[]) or []:
                # If any suggester claims this method is a handler, verify annotation
                claimed = [s for s in group if s.get("class_name")==class_name and s.get("method")==m.name]
                if not claimed: 
                    continue

                verb=None; m_paths=[]
                for a in getattr(m,"annotations",[]) or []:
                    nm=_ann_name(a)
                    if nm in HTTP_SIMPLE:
                        verb = HTTP_SIMPLE[nm]
                        m_paths = _paths_from_vals(_ann_vals(a))
                        break

                # Handler if class or method mapping found; otherwise accept Suggester's 'ANY' only if method looks public
                if not verb and not m_paths and not class_map_paths:
                    # best-effort acceptance if suggester is confident AND method is public
                    if "public" not in (m.modifiers or set()):
                        continue
                    verb = claimed[0].get("http_method") or "ANY"
                    m_paths = claimed[0].get("paths") or [""]

                # Compose full paths
                paths = []
                if class_map_paths and m_paths:
                    for a in class_map_paths:
                        for b in m_paths:
                            paths.append(("/"+"/".join([a.strip("/"), b.strip("/")])).replace("//","/"))
                else:
                    paths = m_paths or class_map_paths or [""]

                params = [p.type.name if p.type else "?" for p in m.parameters]
                fqmn = f"{fqcn}#{m.name}({','.join(params)})"
                verified.append(VerifiedHandler(fqmn=fqmn, http_method=verb or "ANY", paths=paths, file=file_path))

    # De-dup
    uniq = {}
    for v in verified:
        key = (v.fqmn, tuple(v.paths))
        uniq[key]=v
    return list(uniq.values())

# ============= Call Graph Builder (depth-limited) ==========
def _walk_calls(method_node) -> List[Tuple[Optional[str], str]]:
    calls=[]
    for _, node in method_node:
        k=node.__class__.__name__
        if k=="MethodInvocation":
            calls.append((getattr(node,"qualifier",None), getattr(node,"member",None)))
        elif k=="SuperMethodInvocation":
            calls.append(("super", getattr(node,"member",None)))
    return [(q,m) for (q,m) in calls if m]

def _resolve_qualifier(qual: Optional[str], file_path: str, symbol_index: Dict[str,dict]) -> Optional[str]:
    if not qual or qual in ("this","super"): return None
    # naive: match by simple class name in symbol index
    for k, rec in symbol_index.items():
        fqcn = f"{rec['pkg']+'.' if rec['pkg'] else ''}{rec['type']}"
        if fqcn.split(".")[-1]==qual: return fqcn
    return None

def build_symbol_index(root: str) -> Dict[str,dict]:
    idx={}
    for p in glob.glob(os.path.join(root,"**","*.java"), recursive=True):
        try:
            tree=javalang.parse.parse(open(p,"r",encoding="utf-8",errors="ignore").read())
        except Exception:
            continue
        pkg=getattr(tree.package,"name",None)
        for t in getattr(tree,"types",[]) or []:
            if not hasattr(t,"name"): continue
            fqcn=f"{pkg+'.' if pkg else ''}{t.name}"
            for m in getattr(t,"methods",[]) or []:
                key=f"{fqcn}#{m.name}"
                idx[key]={"pkg":pkg,"type":t.name,"method":m.name,"node":m,"file":p}
    return idx

def build_graph(handlers: List[VerifiedHandler], root: str, max_depth:int=2, include_pkg_prefix:Optional[str]=None) -> Dict[str,Any]:
    G=nx.DiGraph()
    symbol_index = build_symbol_index(root)

    def allowed(fqcn:str)->bool:
        return True if not include_pkg_prefix else fqcn.startswith(include_pkg_prefix)

    seeds=[h.fqmn for h in handlers]
    for s in seeds: G.add_node(s, kind="controller")

    frontier=[(s,0) for s in seeds]
    seen=set()
    while frontier:
        cur, d = frontier.pop(0)
        if cur in seen or d>=max_depth:
            seen.add(cur); continue
        seen.add(cur)

        base = cur.split("#")[0]+"#"+cur.split("#")[1].split("(")[0]
        rec = symbol_index.get(base)
        if not rec: continue
        for qual, name in _walk_calls(rec["node"]):
            cands=[]
            fq = _resolve_qualifier(qual, rec["file"], symbol_index)
            if fq:
                for k, r in symbol_index.items():
                    rfqcn=f"{r['pkg']+'.' if r['pkg'] else ''}{r['type']}"
                    if rfqcn==fq and r["method"]==name:
                        cands.append(k)
            else:
                cands=[k for k,r in symbol_index.items() if r["method"]==name]

            for callee in cands:
                callee_fqcn = callee.split("#")[0]
                if not allowed(callee_fqcn): continue
                if callee not in G: G.add_node(callee, kind="")
                G.add_edge(cur, callee)
                if d+1 < max_depth:
                    frontier.append((callee, d+1))

    return {
        "nodes":[{"id":n,"kind":G.nodes[n].get("kind","")} for n in G.nodes()],
        "edges":[{"from":u,"to":v} for u,v in G.edges()],
        "stats":{"nodes":G.number_of_nodes(),"edges":G.number_of_edges(),"max_depth":max_depth}
    }

# ================== ADK Orchestration ======================
class InputConfig(BaseModel):
    repo_root: str = Field(description="Path to repo root")
    include_pkg_prefix: Optional[str] = Field(default=None)
    max_depth: conint(ge=1, le=6) = 2
    export_dir: Optional[str] = None

class LlmPlanner(BaseAgent):
    """LLM-first discovery + deterministic verification + graphing."""
    def __init__(self, name="LlmPlanner"):
        super().__init__(name=name)
        self.fs = FsTool()

    async def _run_async_impl(self, ctx):
        # Expect the user to provide JSON InputConfig in the very first message
        last = ctx.last_user_message
        try:
            cfg = InputConfig.model_validate_json(last.text)
        except Exception:
            tmpl = '{"repo_root":"/abs/path/to/repo","include_pkg_prefix":"com.myco","max_depth":2,"export_dir":"/tmp/out"}'
            yield Event.final_response(f"Please send a JSON config like:\n{tmpl}")
            return

        # 1) Gather samples for the LLM
        java_samples = self.fs.sample_java_files(cfg.repo_root, max_files=40)
        yml_samples = self.fs.sample_yml(cfg.repo_root, max_files=6)

        # Build the LLM input object
        sugg_in = SuggesterInput(repo_root=cfg.repo_root, max_depth=cfg.max_depth, include_pkg_prefix=cfg.include_pkg_prefix)
        parts = [
            types.Part(text="### application.yml samples ###"),
            types.Part(text=json.dumps(yml_samples, indent=2)),
            types.Part(text="\n### java file heads ###"),
            types.Part(text=json.dumps(java_samples[:20], indent=2)),  # keep prompt size reasonable
            types.Part(text="\n### Instruction payload (schema) ###"),
            types.Part(text=sugg_in.model_dump_json()),
        ]
        user_msg = types.Content(role="user", parts=parts)

        # 2) Call the Suggester LLM agent
        async for ev in controller_suggester.run_async(ctx, new_message=user_msg):
            yield ev  # surface intermediate thinking/results

        raw = ctx.session.state.get("controller_suggestions_json")
        if not raw:
            yield Event.final_response("No suggestions produced by LLM.")
            return

        suggestions = SuggesterOutput.model_validate_json(raw).suggestions

        # 3) Deterministic verify/normalize
        verified = verify_and_normalize(cfg.repo_root, suggestions)

        # 4) Build depth-limited graph from verified controller methods
        graph = build_graph(verified, cfg.repo_root, max_depth=cfg.max_depth, include_pkg_prefix=cfg.include_pkg_prefix)

        # 5) Optional exports
        if cfg.export_dir:
            os.makedirs(cfg.export_dir, exist_ok=True)
            # controllers.csv
            with open(os.path.join(cfg.export_dir,"controllers.csv"),"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["fqmn","verb","path","file"])
                for h in verified:
                    for p in h.paths:
                        w.writerow([h.fqmn, h.http_method, p, os.path.relpath(h.file, cfg.repo_root)])
            # call_graph.csv
            with open(os.path.join(cfg.export_dir,"call_graph.csv"),"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["caller","callee"])
                for e in graph["edges"]:
                    w.writerow([e["from"], e["to"]])
            # dot
            with open(os.path.join(cfg.export_dir,"call_graph.dot"),"w",encoding="utf-8") as f:
                f.write("digraph G {\n")
                for e in graph["edges"]:
                    f.write(f"  \"{e['from']}\" -> \"{e['to']}\";\n")
                f.write("}\n")

        # 6) Final summary
        out = {
            "suggested_count": len(suggestions),
            "verified_count": len(verified),
            "controllers": [{"fqmn":h.fqmn,"verb":h.http_method,"paths":h.paths,"file":os.path.relpath(h.file,cfg.repo_root)} for h in verified],
            "graph": graph
        }
        yield Event.final_response("Discovery (LLM-first) result:\n```json\n"+json.dumps(out, indent=2)+"\n```")

# Compose root agent
root = SequentialAgent(name="LLMIntelligentControllers", sub_agents=[LlmPlanner()])

# Local runner
if __name__ == "__main__":
    APP="llm-intel-ctrl"; USER="local"; SID="s1"
    svc = InMemorySessionService()
    runner = Runner(agent=root, app_name=APP, session_service=svc)
    svc.create_session(APP, USER, SID)
    print("Paste JSON like:\n{\"repo_root\":\"/abs/path/to/repo\",\"include_pkg_prefix\":\"com.acme\",\"max_depth\":2,\"export_dir\":\"/tmp/out\"}")
    while True:
        try:
            line = input("\nYou> ").strip()
            if not line: continue
            msg = types.Content(role="user", parts=[types.Part(text=line)])
            for ev in runner.run(user_id=USER, session_id=SID, new_message=msg):
                if ev.is_final_response():
                    print("\nAgent>\n", ev.stringify_content())
        except KeyboardInterrupt:
            break
