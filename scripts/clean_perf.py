import os

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if '[PERF]' in line: continue
        if 't0 =' in line: continue
        if 't0_' in line: continue
        if 't_ser =' in line: continue
        new_lines.append(line)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

clean_file('backend/app/agent.py')
clean_file('backend/app/main.py')
