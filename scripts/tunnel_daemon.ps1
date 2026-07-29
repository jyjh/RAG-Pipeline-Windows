<#
.SYNOPSIS
    Cross-platform auto-reconnecting SSH tunnel daemon for Ollama port forwarding.

.DESCRIPTION
    Establishes an SSH port-forwarding tunnel connecting a local port to a remote HPC host.
    Monitors the background SSH process and automatically reconnects upon disconnection.

    Two modes:
    1. Single-hop (default): forward straight to a host that runs Ollama itself
       (e.g. a login/dev node).
    2. Two-hop (GPU compute node): pass -JumpHost <login> -HostFile <path>. The
       daemon sshes the jump host to read the live compute-node name from
       -HostFile (written by scripts/nus_hpc_serve.pbs), then forwards to it via
       ssh ProxyJump (-J). Re-reads the file on every reconnect so a rescheduled
       serving job on a different node is picked up automatically.

.PARAMETER HpcUser
    Remote HPC SSH username. Defaults to $env:HPC_USER or current logged-in user.

.PARAMETER HpcHost
    Remote HPC target host/hostname alias. Defaults to 'nus_hpc_gpu'.

.PARAMETER LocalPort
    Local listening port to bind for tunneling. Defaults to 11434.

.PARAMETER RemotePort
    Remote target port on the HPC host. Defaults to 11434.

.PARAMETER SshKey
    Optional path to the SSH private key file.

.PARAMETER JumpHost
    Login/jump host for 2-hop mode (ProxyJump). Enables two-hop forwarding to a
    GPU compute node that runs Ollama inside scripts/nus_hpc_serve.pbs.

.PARAMETER HostFile
    Discovery file (on the jump host) written by the serving PBS job, containing
    the live compute hostname. Recomputed on each reconnect. Requires -JumpHost.

.PARAMETER ReconnectInterval
    Delay in seconds between reconnection attempts. Defaults to 5.

.EXAMPLE
    .\scripts\tunnel_daemon.ps1 -HpcUser "e0123456" -HpcHost "nus_hpc_gpu"

.EXAMPLE
    .\scripts\tunnel_daemon.ps1 -LocalPort 11434 -RemotePort 11434 -SshKey "$HOME\.ssh\id_rsa"

.EXAMPLE
    .\scripts\tunnel_daemon.ps1 -JumpHost "nus_hpc" -HostFile "$HOME\.rag_ollama_serving_host"
#>

[CmdletBinding()]
param(
    [string]$HpcUser = $env:HPC_USER,
    [string]$HpcHost = $(if ($env:HPC_HOST) { $env:HPC_HOST } else { "nus_hpc_gpu" }),
    [int]$LocalPort = $(if ($env:LOCAL_PORT) { [int]$env:LOCAL_PORT } else { 11434 }),
    [int]$RemotePort = $(if ($env:REMOTE_PORT) { [int]$env:REMOTE_PORT } else { 11434 }),
    [string]$SshKey = $env:SSH_KEY,
    [string]$JumpHost = $env:JUMP_HOST,
    [string]$HostFile = $env:HOST_FILE,
    [int]$ReconnectInterval = 5
)

if ([string]::IsNullOrWhiteSpace($HpcUser)) {
    if ($env:USERNAME) {
        $HpcUser = $env:USERNAME
    } elseif ($env:USER) {
        $HpcUser = $env:USER
    } else {
        $HpcUser = [Environment]::UserName
    }
}

function Get-TunnelTarget {
    # In 2-hop mode, ssh the jump host to read the discovery file (written by
    # scripts/nus_hpc_serve.pbs). In single-hop mode, just return $HpcHost.
    if (-not [string]::IsNullOrWhiteSpace($JumpHost)) {
        if ([string]::IsNullOrWhiteSpace($HostFile)) {
            throw "-HostFile is required when -JumpHost is set"
        }
        $probeArgs = @("-o", "ExitOnForwardFailure=no", $JumpHost, "cat `"$HostFile`"")
        $discovered = (& ssh.exe @probeArgs 2>$null | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($discovered)) {
            throw "Discovery file '$HostFile' on '$JumpHost' is empty or missing (is scripts/nus_hpc_serve.pbs running?)"
        }
        return $discovered
    }
    return $HpcHost
}

Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Starting SSH tunnel daemon..."
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] User: ${HpcUser}"
if (-not [string]::IsNullOrWhiteSpace($JumpHost)) {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Mode: 2-hop (jump='${JumpHost}', host-file='${HostFile}')"
} else {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Mode: single-hop (target='${HpcHost}')"
}
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Port Forwarding: ${LocalPort} -> localhost:${RemotePort}"

try {
    while ($true) {
        $timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')

        try {
            $target = Get-TunnelTarget
        } catch {
            Write-Host "[$timestamp] $_ Reconnecting in ${ReconnectInterval}s..."
            Start-Sleep -Seconds $ReconnectInterval
            continue
        }

        Write-Host "[$timestamp] Establishing SSH tunnel to ${HpcUser}@${target}..."

        $sshArgs = @(
            "-N",
            "-T",
            "-L", "${LocalPort}:localhost:${RemotePort}",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3"
        )

        if (-not [string]::IsNullOrWhiteSpace($JumpHost)) {
            $sshArgs += "-J"
            $sshArgs += $JumpHost
        }
        if (-not [string]::IsNullOrWhiteSpace($SshKey)) {
            $sshArgs += "-i"
            $sshArgs += $SshKey
        }
        $sshArgs += "${HpcUser}@${target}"

        $process = Start-Process -FilePath "ssh.exe" -ArgumentList $sshArgs -NoNewWindow -PassThru

        try {
            while (-not $process.HasExited) {
                Start-Sleep -Milliseconds 500
            }
        } finally {
            if ($process -and -not $process.HasExited) {
                Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Stopping SSH process (PID: $($process.Id))..."
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }

        Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Tunnel disconnected. Reconnecting in ${ReconnectInterval}s..."
        Start-Sleep -Seconds $ReconnectInterval
    }
} finally {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] SSH tunnel daemon stopped."
}
