import re

with open('backend/app/agent.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix unindented returns
code = code.replace(
    '            trace.append(f"INPUT_VALIDATION FAILED: {f[\'filename\']} -> {res.reason}")\n    return {"api_response": {"status": "INVALID_INPUT"',
    '            trace.append(f"INPUT_VALIDATION FAILED: {f[\'filename\']} -> {res.reason}")\n            return {"api_response": {"status": "INVALID_INPUT"'
)

code = code.replace(
    '            trace.append(f"INPUT_VALIDATION FAILED (PAIR): {pair_res.reason}")\n    return {"api_response": {"status": "DATA_UNAVAILABLE"',
    '            trace.append(f"INPUT_VALIDATION FAILED (PAIR): {pair_res.reason}")\n            return {"api_response": {"status": "DATA_UNAVAILABLE"'
)

code = code.replace(
    '    return {"parsed_intent": intent, "trace": trace}\n',
    '    return {"parsed_intent": intent, "trace": trace}\n'
)

# wait there are multiple places where I stripped the indent when I replaced `return {`
# Let's just fix all `    return {` that should be `        return {`? No, some should be `    return {`.
