import subprocess
import sys

def main():
    cmd = [
        sys.executable,   # current python interpreter
        "-m", "mlgame3d",
        "-i", "mlplay.py",
        "-i", "hidden",
        "-i", "hidden",
        "-i", "hidden",
        "-e", "10000",
        "-gp", "audio", "false",
        # "-gp", "max_time", "5", # 5 minutes
        "-gp", "map", "4",
        "-ts", "5",
        "--fps", "60",
        "proly.exe"
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
