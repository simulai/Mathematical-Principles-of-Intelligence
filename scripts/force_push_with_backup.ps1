<#
Usage: run from repo root in PowerShell
.
.
Example: 
  .\scripts\force_push_with_backup.ps1 -Remote origin -Branch main

This script fetches the remote, creates a remote backup branch named
`backup/<branch>-YYYYMMDD-HHMMSS` that points to the current remote branch tip,
then force-pushes the local branch to remote.
#>
param(
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

function ExecGit {
    param($args)
    Write-Host "git $args"
    $output = git $args 2>&1
    $exit = $LASTEXITCODE
    if ($output) { Write-Host $output }
    if ($exit -ne 0) { throw "git $args failed (exit $exit)" }
    return $output
}

try {
    Push-Location -LiteralPath (Get-Location)

    ExecGit "fetch $Remote --quiet"

    $remoteRef = "$Remote/$Branch"
    $hash = (git rev-parse $remoteRef) -replace "\r|\n",""
    if (-not $hash) { throw "Failed to resolve remote ref: $remoteRef" }

    $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $backupBranch = "backup/$Branch-$timestamp"

    # create local branch pointing at remote hash (force if exists)
    ExecGit "branch -f $backupBranch $hash"

    # push the backup branch to remote
    ExecGit "push $Remote $backupBranch"

    # force-push the branch
    ExecGit "push --force $Remote $Branch"

    Write-Host "Force-push completed. Backup branch pushed as: $Remote/$backupBranch"
} catch {
    Write-Error "Error: $_"
    exit 1
} finally {
    Pop-Location
}
