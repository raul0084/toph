# toph GPU runner for Windows. Paste this into PowerShell:
#
#   irm https://raw.githubusercontent.com/raul0084/toph/main/scripts/gpu/windows.ps1 | iex
#
# What it does:
#   1. Checks for an NVIDIA GPU.
#   2. Installs WSL2 + Ubuntu if missing (asks for admin once, then needs ONE reboot;
#      after rebooting, paste the same line again).
#   3. Runs setup.sh inside Ubuntu, which does all the real work.
#   4. Copies the results zip to your Desktop.

$ErrorActionPreference = 'Stop'
$Distro = 'Ubuntu'
$SetupUrl = 'https://raw.githubusercontent.com/raul0084/toph/main/scripts/gpu/setup.sh'

function Say($msg, $color = 'Cyan') { Write-Host "`n==> $msg" -ForegroundColor $color }

# 1. GPU driver
Say 'Checking for an NVIDIA GPU'
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Say 'No NVIDIA driver found. Install the latest driver from https://www.nvidia.com/Download/index.aspx, reboot, then run this again.' 'Red'
    return
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# 2. WSL + Ubuntu
$distros = ''
try { $distros = (wsl.exe -l -q 2>$null) -join "`n" -replace "`0", '' } catch {}
if ($distros -notmatch "(?m)^$Distro" -and (Get-Command ubuntu.exe -ErrorAction SilentlyContinue)) {
    # Installed after reboot but never launched: register it with root as the default user,
    # skipping the interactive username/password prompt.
    Say 'Finishing Ubuntu setup'
    ubuntu.exe install --root
    $distros = (wsl.exe -l -q 2>$null) -join "`n" -replace "`0", ''
}
if ($distros -notmatch "(?m)^$Distro") {
    Say 'Installing WSL2 + Ubuntu. A Windows admin prompt will appear - click Yes.'
    $p = Start-Process wsl.exe -ArgumentList "--install -d $Distro --no-launch" -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        Say "WSL install failed (code $($p.ExitCode)). Virtualization may be disabled in the BIOS (look for 'SVM' on AMD or 'VT-x' on Intel)." 'Red'
        return
    }
    Say 'WSL installed. Please REBOOT the computer, then paste the same command again.' 'Yellow'
    return
}

# 3. Run the real setup inside Ubuntu as root (avoids the Ubuntu username/password prompt).
Say 'Running setup inside Ubuntu. This can take a few hours (big download + processing). Leave this window open.'
wsl.exe -d $Distro -u root -- bash -lc "curl -fsSL $SetupUrl | bash"
if ($LASTEXITCODE -ne 0) {
    $log = "\\wsl.localhost\$Distro\root\toph-gpu\setup.log"
    $desk = [Environment]::GetFolderPath('Desktop')
    if (Test-Path $log) { Copy-Item $log $desk -Force }
    Say "Setup failed. A file called setup.log was put on your Desktop - please send it back. Re-running this command resumes where it stopped." 'Red'
    return
}

# 4. Results to Desktop
$desk = [Environment]::GetFolderPath('Desktop')
Get-ChildItem "\\wsl.localhost\$Distro\root\toph-gpu\toph_detections_*.zip" | Copy-Item -Destination $desk -Force
Say "Done! The results zip is on your Desktop ($desk). Please send it back." 'Green'
