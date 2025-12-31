Force-push helper

Usage:

PowerShell (from repository root):

```powershell
.
\scripts\force_push_with_backup.ps1 -Remote origin -Branch main
```

What it does:
- Fetches `origin`
- Creates a backup branch on the remote named `backup/<branch>-YYYYMMDD-HHMMSS` pointing to the remote's current tip
- Force-pushes the local branch to the remote

Notes:
- The script assumes `git` is available in PATH and you have network access and permissions to push.
- Use responsibly: force-pushing rewrites remote history — backups are created by this script before overwrite.
