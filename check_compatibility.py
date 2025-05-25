#!/usr/bin/env python3
"""Check AIDER v0.83.1 compatibility"""

import sys
sys.path.insert(0, 'src')

import inspect
from aider.coders import Coder
import aider

print('AIDER VERSION:', getattr(aider, '__version__', 'unknown'))

# Check Coder.__init__ signature
init_sig = inspect.signature(Coder.__init__)
print('\nCoder.__init__ parameters:')
for param_name, param in init_sig.parameters.items():
    if param_name != 'self':
        default = 'REQUIRED' if param.default == param.empty else repr(param.default)
        print(f'  - {param_name}: {default}')

# Check Coder.create signature
if hasattr(Coder, 'create'):
    create_sig = inspect.signature(Coder.create)
    print('\nCoder.create parameters:')
    for param_name, param in create_sig.parameters.items():
        if param_name != 'cls':
            default = 'REQUIRED' if param.default == param.empty else repr(param.default)
            print(f'  - {param_name}: {default}')
else:
    print('\nCoder.create method not found')

# Check what parameters our MCP server is trying to use
print('\n=== MCP SERVER PARAMETER USAGE ===')
print('Create params we use:')
print('  - main_model')
print('  - io') 
print('  - edit_format')

print('\nInit params we use:')
mcp_init_params = [
    'fnames', 'read_only_fnames', 'repo', 'show_diffs', 
    'auto_commits', 'dirty_commits', 'use_git', 'stream',
    'suggest_shell_commands', 'detect_urls', 'verbose', 'auto_accept_architect'
]
for param in mcp_init_params:
    print(f'  - {param}')