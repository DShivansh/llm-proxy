from __future__ import annotations

import uvicorn

from llm_proxy.app import create_app
from llm_proxy.config import load_config


config = load_config()
app = create_app(config)


def main() -> None:
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
