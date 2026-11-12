# Running the detector (for the GPU machine)

You need: an NVIDIA graphics card, ~90 GB free disk, and a few hours of leaving the computer on.

## Windows

1. Open the Start menu, type **PowerShell**, open it.
2. Paste this line and press Enter:

   ```powershell
   irm https://raw.githubusercontent.com/raul0084/toph/main/scripts/gpu/windows.ps1 | iex
   ```

3. The first time, it installs Ubuntu (click **Yes** on the admin prompt) and asks you to **reboot**.
   After rebooting, paste the same line again.
4. When it says **Done**, there's a `toph_detections_....zip` on your Desktop. Send it back.

## Linux

```bash
curl -fsSL https://raw.githubusercontent.com/raul0084/toph/main/scripts/gpu/setup.sh | bash
```

The zip ends up in `~/toph-gpu/`.

## If something fails

Re-running the same line continues where it stopped (downloads resume). If it keeps failing,
send back `setup.log` (on the Desktop on Windows, `~/toph-gpu/setup.log` on Linux).

To remove everything afterwards: delete `~/toph-gpu` (on Windows, run `wsl --unregister Ubuntu`
in PowerShell to remove the Ubuntu install entirely).
