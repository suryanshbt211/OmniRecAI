from sqlalchemy import text
from backend.database import engine


def get_item(item_id):

    with engine.connect() as conn:

        result = conn.execute(
            text(
                """
                SELECT *
                FROM items
                WHERE item_id=:id
                """
            ),
            {"id": item_id}
        )

        row = result.fetchone()

        if row:
            return dict(row._mapping)

        return None



def get_items_bulk(item_ids):

    with engine.connect() as conn:

        result = conn.execute(
            text(
                """
                SELECT *
                FROM items
                WHERE item_id = ANY(:ids)
                """
            ),
            {
                "ids": item_ids
            }
        )

        return [
            dict(row._mapping)
            for row in result
        ]