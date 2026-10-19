#!/usr/bin/env bash
# Mount the project volume on a fresh E2E node (docs/rental-setup.md, section 2c).
#   sudo bash scripts/e2e_attach.sh              # finds the attached, unformatted-or-labelled disk
#   sudo bash scripts/e2e_attach.sh /dev/vdb     # or name the device (lsblk shows it)
# First use formats the disk as ext4 with the label "rlvr" after you type YES; later uses only
# mount it. The mount point is /mnt/rlvr (override with RLVR_MOUNT).
set -euo pipefail
MOUNT="${RLVR_MOUNT:-/mnt/rlvr}"
DEV="${1:-}"

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo"; exit 1
fi

if [ -z "$DEV" ]; then
  # Prefer a disk already labelled rlvr; otherwise the largest whole disk with no partitions,
  # no filesystem and no mount (the freshly attached volume). The root disk has partitions.
  DEV="$(blkid -L rlvr 2>/dev/null || true)"
fi
if [ -z "$DEV" ]; then
  DEV="$(python3 - <<'PY'
import json, subprocess
out = subprocess.check_output(["lsblk", "-J", "-b", "-o", "NAME,TYPE,MOUNTPOINTS,FSTYPE,SIZE,PATH"])
devs = json.loads(out)["blockdevices"]
cands = [
    d for d in devs
    if d["type"] == "disk" and not d.get("children") and not d.get("fstype")
    and not any(d.get("mountpoints") or [])
]
cands.sort(key=lambda d: -int(d["size"]))
print(cands[0]["path"] if cands else "")
PY
)"
fi
if [ -z "$DEV" ]; then
  echo "no candidate disk found. Attach the volume in MyAccount, then run  lsblk  and pass the device."
  exit 1
fi

FSTYPE="$(blkid -o value -s TYPE "$DEV" 2>/dev/null || true)"
if [ -z "$FSTYPE" ]; then
  echo "== $DEV has no filesystem (size $(lsblk -dno SIZE "$DEV")). First use: it will be formatted as ext4."
  echo "   Everything on $DEV is erased. Type YES to continue:"
  read -r ans
  [ "$ans" = "YES" ] || { echo "aborted"; exit 1; }
  mkfs.ext4 -q -L rlvr "$DEV"
elif [ "$FSTYPE" != "ext4" ]; then
  echo "$DEV has a $FSTYPE filesystem; refusing to touch it"; exit 1
fi

mkdir -p "$MOUNT"
if ! mountpoint -q "$MOUNT"; then
  mount "$DEV" "$MOUNT"
fi
if blkid -L rlvr >/dev/null 2>&1 && ! grep -q " $MOUNT " /etc/fstab; then
  echo "LABEL=rlvr $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
fi
if [ -n "${SUDO_USER:-}" ]; then
  chown "$SUDO_USER":"$SUDO_USER" "$MOUNT"
fi
echo "== $DEV mounted at $MOUNT"
df -h "$MOUNT" | tail -1
echo
echo "next (as your user, not root):"
echo "  cd $MOUNT/reasoning-rlvr && source scripts/e2e_env.sh && git pull"
echo "  (first time: git clone https://github.com/rrtchaganti/reasoning-rlvr.git $MOUNT/reasoning-rlvr)"
