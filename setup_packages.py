import os
import sys
import subprocess

# ==============================================================================
# ملف تثبيت وتجهيز المكتبات لاستضافة bot-hosting.net
# يتجاوز حظر الصلاحيات على /home/container/.local ويثبت المكتبات في المسار المسموح
# ==============================================================================
CONTAINER_ROOT = "/home/container"
PACKAGE_DIR = "/home/container/packages" if os.path.exists(CONTAINER_ROOT) else os.path.abspath("./packages")

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "websockets",
    "wsproto",
    "httpx",
    "numpy",
]

def install_dependencies():
    os.makedirs(PACKAGE_DIR, exist_ok=True)
    if PACKAGE_DIR not in sys.path:
        sys.path.insert(0, PACKAGE_DIR)
    if os.path.exists(CONTAINER_ROOT) and CONTAINER_ROOT not in sys.path:
        sys.path.insert(0, CONTAINER_ROOT)

    os.environ["PYTHONPATH"] = f"{PACKAGE_DIR}:{CONTAINER_ROOT}:" + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUSERBASE"] = PACKAGE_DIR

    # تحديث ملف requirements.txt تلقائياً إذا كانت لوحة التحكم تقرؤه
    try:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(REQUIRED_PACKAGES) + "\n")
    except Exception:
        pass

    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[SETUP] جاري تثبيت المكتبات الناقصة في {PACKAGE_DIR}: {missing}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--upgrade",
            f"--target={PACKAGE_DIR}",
            *missing
        ])
        print("[SETUP] تم تثبيت جميع المكتبات بنجاح!")
    else:
        print("[SETUP] جميع المكتبات مثبتة وجاهزة للعمل.")

if __name__ == "__main__":
    install_dependencies()
