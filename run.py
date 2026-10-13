"""Start the web app:  python run.py  ->  http://localhost:8000"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
