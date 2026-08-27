import sys
for f in ['backend/app/local_specialists.py', 'backend/app/change_map.py', 'backend/app/optical_sar.py']:
    content = open(f, 'r').read()
    content = content.replace('"input_filenames":', '"device": device if "device" in globals() or "device" in locals() else "cpu", "input_filenames":')
    open(f, 'w').write(content)
