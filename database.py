## datbase creation
from sqlalchemy import create_engine,exc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


SQL_ALCHEMY_DATABASE_URL = "sqlite+pysqlite:///./heart_disease.db"
connect_args = {"check_same_thread": False}
engine = create_engine(SQL_ALCHEMY_DATABASE_URL, connect_args=connect_args)


SessionLocal = sessionmaker(bind = engine,autocommit=False,autoflush=False)

Base = declarative_base()
