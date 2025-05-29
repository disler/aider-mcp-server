#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Find the installed aider-mcp-server package
function findAiderMcpServer() {
    // First try using the npm package location
    const packageDir = path.resolve(__dirname, '..');
    
    if (fs.existsSync(path.join(packageDir, 'src', 'aider_mcp_server'))) {
        return packageDir;
    }
    
    // Try the current working directory
    const cwd = process.cwd();
    if (fs.existsSync(path.join(cwd, 'src', 'aider_mcp_server'))) {
        return cwd;
    }
    
    // Try the parent directory
    const parentDir = path.dirname(cwd);
    if (fs.existsSync(path.join(parentDir, 'src', 'aider_mcp_server'))) {
        return parentDir;
    }
    
    console.error('Error: Could not find aider-mcp-server installation.');
    console.error('Please run this command from the aider-mcp-server directory or specify --aider-dir.');
    process.exit(1);
}

// Main function
function main() {
    const aiderDir = findAiderMcpServer();
    const scriptPath = path.join(aiderDir, 'src', 'aider_mcp_server', 'setup_aider_mcp.py');
    
    // Check if the Python script exists
    if (!fs.existsSync(scriptPath)) {
        console.error(`Error: Setup script not found at ${scriptPath}`);
        process.exit(1);
    }
    
    // Launch the Python script with the same arguments
    const args = ['--directory', aiderDir, 'run', 'python3', scriptPath, ...process.argv.slice(2)];
    
    console.log(`Running setup script from: ${aiderDir}`);
    
    const pythonProcess = spawn('uv', args, { 
        stdio: 'inherit',
        env: process.env 
    });
    
    pythonProcess.on('error', (err) => {
        console.error(`Failed to start Python script: ${err.message}`);
        process.exit(1);
    });
    
    pythonProcess.on('close', (code) => {
        process.exit(code);
    });
}

// Run the main function
main();