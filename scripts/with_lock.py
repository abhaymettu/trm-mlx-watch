"""Run a command while holding /tmp/overnight-compute.lock (the overnight one-heavy-job rule).

  python3 scripts/with_lock.py [--wait] -- CMD ...

macOS has no flock(1), so this takes the same BSD flock lock through fcntl. Without --wait it exits 75
when another job holds the lock, so the caller can do prep work and retry.
"""
import fcntl
import os
import subprocess
import sys

LOCK = "/tmp/overnight-compute.lock"


def main(argv):
    wait = "--wait" in argv[: argv.index("--")] if "--" in argv else False
    cmd = argv[argv.index("--") + 1:] if "--" in argv else argv
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        print("compute lock held by another job: %s" % LOCK, file=sys.stderr)
        return 75
    os.ftruncate(fd, 0)
    os.write(fd, ("%d %s\n" % (os.getpid(), " ".join(cmd)[:200])).encode())
    try:
        return subprocess.call(cmd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
