import os

DEV = "development"
PROD = "production"


class MyEnv:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MyEnv, cls).__new__(cls)
            cls._instance.port = int(os.getenv("API_PORT", 3001))
            cls._instance.host = os.getenv("API_HOST", "localhost")
            cls._instance.env = os.getenv("API_ENV", DEV)
        return cls._instance

    def is_dev(self) -> bool:
        return self.env == DEV

    def __str__(self) -> str:
        return f"MyEnv(port={self.port}, host='{self.host}', env='{self.env}')"


my_env = MyEnv()
