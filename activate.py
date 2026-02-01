import subprocess
import sys

def main():
    cmd = [
        sys.executable,   # current python interpreter
        "-m", "mlgame3d",
        "-i", "mlplay.py",
        "proly.exe"
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
