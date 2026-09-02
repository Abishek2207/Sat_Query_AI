import re
with open('backend/app/model_registry.py', 'r') as f:
    text = f.read()
text = text.replace('"local"', '""')
with open('backend/app/model_registry.py', 'w') as f:
    f.write(text)
