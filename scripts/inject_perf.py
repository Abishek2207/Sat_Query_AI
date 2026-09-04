import os
import re

path = 'backend/app/agent.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

if 'import time' not in code:
    code = 'import time\n' + code

# validate_node
code = re.sub(r'def validate_node\(state: AgentState\):\n(\s+)trace =', 
              r'def validate_node(state: AgentState):\n\1t_val = time.time()\n\1trace =', code)
code = re.sub(r'(return \{"api_response": \{"status": "DATA_UNAVAILABLE"[\s\S]*?\})',
              r'print(f"[PERF] file validation: {time.time()-t_val:.2f}s")\n        \1', code)
code = re.sub(r'trace.append\("INPUT_VALIDATION: SUCCESS"\)\n(\s+)return ',
              r'trace.append("INPUT_VALIDATION: SUCCESS")\n\1print(f"[PERF] file validation: {time.time()-t_val:.2f}s")\n\1return ', code)

# parse_query
code = re.sub(r'def parse_query_node\(state: AgentState\):\n(\s+)if state',
              r'def parse_query_node(state: AgentState):\n\1t_parse = time.time()\n\1if state', code)
code = re.sub(r'(return \{"parsed_intent": intent, "trace": trace\})',
              r'print(f"[PERF] query parsing: {time.time()-t_parse:.2f}s")\n    \1', code)

# plan_tools
code = re.sub(r'def plan_tools_node\(state: AgentState\):\n(\s+)if state',
              r'def plan_tools_node(state: AgentState):\n\1t_plan = time.time()\n\1if state', code)
code = re.sub(r'(return \{"selected_tools": tools, "trace": trace, "warnings": warnings\})',
              r'print(f"[PERF] specialist selection: {time.time()-t_plan:.2f}s")\n    \1', code)

# execute_tools
code = re.sub(r'def execute_tools_node\(state: AgentState\):\n(\s+)if state',
              r'def execute_tools_node(state: AgentState):\n\1t_exec = time.time()\n\1if state', code)
code = re.sub(r'(return \{"tool_results": tool_results, "trace": trace\})',
              r'print(f"[PERF] specialist inference: {time.time()-t_exec:.2f}s")\n    \1', code)

# verify_evidence
code = re.sub(r'def verify_evidence_node\(state: AgentState\):\n(\s+)if state',
              r'def verify_evidence_node(state: AgentState):\n\1t_ver = time.time()\n\1if state', code)
code = re.sub(r'trace.append\("EVIDENCE_CHECK: SUCCESS"\)\n(\s+)return ',
              r'trace.append("EVIDENCE_CHECK: SUCCESS")\n\1print(f"[PERF] evidence verification: {time.time()-t_ver:.2f}s")\n\1return ', code)

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

path = 'backend/app/main.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

if 'import time' not in code:
    code = 'import time\n' + code

code = re.sub(r'(response_obj = AnalysisResponse\()', r't_ser = time.time()\n        \1', code)
code = re.sub(r'(return response_obj)', r'print(f"[PERF] response serialization: {time.time()-t_ser:.2f}s")\n        \1', code)

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)
