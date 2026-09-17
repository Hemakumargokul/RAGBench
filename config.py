from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_ENV:str = "dev"
    APP_HOST:str = "0.0.0.0"
    APP_PORT:int = 8000
    OPENAI_API_KEY: str = ""
    DOCUMENT_AI_PROJECT_ID: str = "541853870336"
    DOCUMENT_AI_LOCATION: str = "us"
    DOCUMENT_AI_PROCESSOR_ID: str = "557897cc371b8fa1"

    class Config:
        env_file = ".env"

settings = Settings()