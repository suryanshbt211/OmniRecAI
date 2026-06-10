from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String


class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "users"

    user_id = Column(
        Integer,
        primary_key=True
    )

    age = Column(Integer)

    premium_user = Column(Integer)


class Item(Base):

    __tablename__ = "items"

    item_id = Column(
        Integer,
        primary_key=True
    )

    category = Column(String)

    price = Column(Float)

    rating = Column(Float)