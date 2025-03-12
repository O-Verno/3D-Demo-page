import uvicorn

from api.my_env import my_env


def main():
    """Launch the FastAPI application through ASGI Uvicorn"""
    uvicorn.run(
        "api.app:app",
        host=my_env.host,
        port=my_env.port,
        reload=my_env.is_dev()
    )


if __name__ == '__main__':
    main()
