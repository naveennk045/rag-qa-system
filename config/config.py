import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    VECTOR_DB_DIR = os.path.join(BASE_DIR, 'vector_db')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

    # Model Settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1500))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 150))

    # FAISS Settings
    FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, 'faiss_index.bin')
    METADATA_PATH = os.path.join(VECTOR_DB_DIR, 'faiss_metadata.json')
    DOCUMENT_MAPPING_PATH = os.path.join(VECTOR_DB_DIR, 'document_mapping.json')

    # Create directories if they don't exist
    @staticmethod
    def create_directories():
        dirs = [
            Config.DATA_DIR, Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR,
            Config.VECTOR_DB_DIR, Config.MODELS_DIR, Config.LOGS_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
