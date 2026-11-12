# toph

LiDAR 3D multi-object tracking on the KITTI Tracking dataset. A neural network (PV-RCNN, via
OpenPCDet) detects cars, pedestrians and cyclists in each LiDAR scan; a Kalman-filter tracker
then links those detections over time so each object keeps a stable ID.

Detection needs an NVIDIA GPU, so it runs on a separate machine. Everything else (tracking,
evaluation, visualization) runs on a normal laptop.

---

## For Nick: running the detector on your PC

Thanks for helping! This runs a 3D object detector over the KITTI driving dataset on your
graphics card and produces one zip file to send back. You don't need to know anything about
the project; it's a single command, and it cleans up after itself if you want.

### What you need
- A Windows PC with an **NVIDIA graphics card** and an up-to-date driver
  ([nvidia.com/Download](https://www.nvidia.com/Download/index.aspx)).
- About **90 GB of free disk space** (mostly the dataset download, ~35 GB zipped).
- A few hours where the PC can stay on. You can keep using it, but games or other GPU-heavy
  work will slow it down.

### What this does to your PC
Pasting a command from the internet means trusting whoever wrote it, so here's exactly what
it does:
- Installs **WSL + Ubuntu** (Microsoft's official Linux-inside-Windows). Everything else lives
  inside it, in one folder (`~/toph-gpu`); nothing is installed into Windows itself.
- Downloads a private Python, PyTorch, the OpenPCDet detector, a pretrained model and the KITTI
  dataset. The installer and model are checked against fixed SHA-256 fingerprints, and it
  stops if they don't match.
- Runs only the code from a **fixed, frozen version** of this repo (tag `v0.1`), so later
  changes to the repo can't change what runs on your machine. You can read it all at
  [github.com/raul0084/toph/tree/v0.1](https://github.com/raul0084/toph/tree/v0.1).
- Ubuntu is set up to log in as its admin user (root) with no password, to skip the setup
  questions. That's fine on a personal PC, but remove it when you're done (see below).
- The zip you send back contains only detected boxes (text files), timing info, your GPU model
  and a log of the install. No personal files.

### Steps
1. Open the **Start menu**, type **PowerShell**, and open it (no need for "Run as administrator").
2. Paste this line and press **Enter**:

   ```powershell
   irm https://raw.githubusercontent.com/raul0084/toph/v0.1/scripts/gpu/windows.ps1 | iex
   ```

3. **First time only:** it installs Ubuntu (a Linux environment that runs inside Windows, called
   WSL). A Windows admin prompt appears — click **Yes**. When it says so, **reboot**, then open
   PowerShell again and paste the **same line** again.
4. It now downloads the dataset and the model, then runs detection. It prints progress as it
   goes. **Leave the window open.**
5. When it says **Done!**, there's a file called `toph_detections_pv_rcnn.zip` on your
   **Desktop**. Send it back (WeTransfer, Google Drive, etc. — it may be too big for email).

### If something goes wrong
- **Just run the same line again.** It skips every step that already finished and resumes
  downloads where they stopped.
- "No NVIDIA driver found": install the driver from the link above, reboot, re-run.
- "Virtualization may be disabled in the BIOS": WSL needs CPU virtualization turned on
  (called **SVM** on AMD or **VT-x / Intel Virtualization** on Intel, in the BIOS settings).
- "Google Drive refused the download": the model file is rate-limited; try again in a few hours.
- If it keeps failing, a file called `setup.log` is put on your Desktop. Send that instead.

### Removing everything afterwards
In PowerShell, run:

```powershell
wsl --unregister Ubuntu
```

This deletes the Ubuntu install along with the dataset and all files it created.

### On Linux instead of Windows
```bash
curl -fsSL https://raw.githubusercontent.com/raul0084/toph/v0.1/scripts/gpu/setup.sh | bash
```
The zip ends up in `~/toph-gpu/`; delete that folder to uninstall.

---

## For development (Mac side)

```bash
conda activate holocene
python -m pytest                      # unit tests
python scripts/track_gt_sanity.py     # tracker on noisy ground truth, no detector needed
```

See `CLAUDE.md` for conventions and data layout, and `study.md` for background reading.
