import sys
t = sys.stdin.read()
lines = t.splitlines()
keep = ('FAILED', 'tests/', 'E  ', 'E ', '> ', 'assert', 'Error', 'hpc', 'config', '_ _ _', 'short test summary')
out = [l for l in lines if l.startswith(keep)]
print(chr(10).join(out[-60:]))
