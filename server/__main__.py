import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=os.getenv("STUDY_HOST", "127.0.0.1"),
        port=int(os.getenv("STUDY_PORT", "8000")),
        reload=False,
    )
