import re

def add_perf_to_agent():
    with open('backend/app/agent.py', 'r', encoding='utf-8') as f:
        code = f.read()

    if 'import time' not in code:
        code = 'import time\n' + code

    nodes = [
        ("validate_node", "file validation"),
        ("parse_query_node", "query parsing"),
        ("plan_tools_node", "specialist selection"),
        ("execute_tools_node", "specialist inference"),
        ("verify_evidence_node", "evidence verification"),
    ]

    for node_name, label in nodes:
        # Match def node_name(state: AgentState):
        pattern = r'(def ' + node_name + r'\(state: AgentState\):\n)'
        replacement = r'\1    t0_' + node_name + r' = time.time()\n'
        code = re.sub(pattern, replacement, code)
        
        # Match return statements and inject print before them
        # This requires careful regex or just replacing 'return {'
        ret_pattern = r'(return \{)'
        ret_repl = r'print(f"[PERF] ' + label + r': {time.time() - t0_' + node_name + r':.2f}s")\n    \1'
        
        # We only want to replace returns INSIDE this function.
        # It's easier to split by 'def '
        
    # Split by def
    parts = code.split('def ')
    new_parts = [parts[0]]
    for part in parts[1:]:
        for node_name, label in nodes:
            if part.startswith(node_name + '(state: AgentState):'):
                part = part.replace(':\n', ':\n    t0 = time.time()\n', 1)
                part = part.replace('return {', f'print(f"[PERF] {label}: {{time.time() - t0:.2f}}s")\n    return {{')
                part = part.replace('return state', f'print(f"[PERF] {label}: {{time.time() - t0:.2f}}s")\n    return state')
        new_parts.append(part)
        
    code = 'def '.join(new_parts)
    
    with open('backend/app/agent.py', 'w', encoding='utf-8') as f:
        f.write(code)

def add_perf_to_main():
    with open('backend/app/main.py', 'r', encoding='utf-8') as f:
        code = f.read()
    if 'import time' not in code:
        code = 'import time\n' + code
    
    code = code.replace(
        'response_obj = AnalysisResponse(',
        't_ser = time.time()\n        response_obj = AnalysisResponse('
    )
    code = code.replace(
        'return response_obj',
        'print(f"[PERF] response serialization: {time.time() - t_ser:.2f}s")\n        return response_obj'
    )
    
    with open('backend/app/main.py', 'w', encoding='utf-8') as f:
        f.write(code)

add_perf_to_agent()
add_perf_to_main()
