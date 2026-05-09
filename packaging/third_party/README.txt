Tailscale installer (optional bundle for Inno Setup)
======================================================

Place Windows amd64 Tailscale standalone installer next to this file as:

  tailscale-setup-amd64.exe

Download the current stable installer from Tailscale:

  https://pkgs.tailscale.com/stable/tailscale-setup-amd64.exe

Running ../../packaging/build-installer.sh on Linux downloads this file automatically
before invoking ISCC (unless it already exists).

If this file is missing, the RetroIkea Inno Setup script still builds; the Tailscale
install task is skipped (skipifsourcedoesntexist).
