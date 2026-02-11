# Fetch runpodctl.exe
$ErrorActionPreference = "Stop"

$url = "https://github.com/runpod/runpodctl/releases/download/v1.13.0/runpodctl-windows-amd64.exe"
$output = "$PSScriptRoot\..\runpodctl.exe"

Write-Host "Downloading runpodctl from $url..."
Invoke-WebRequest -Uri $url -OutFile $output

if (Test-Path $output) {
    Write-Host "✅ runpodctl.exe downloaded successfully to $output"
    # Unblock the file (Windows security)
    Unblock-File -Path $output
} else {
    Write-Error "❌ Download failed."
}
