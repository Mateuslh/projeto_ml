import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from psycopg2 import pool

load_dotenv()

DB_PARAMS = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# connection_pool = pool.SimpleConnectionPool(
#     1, 20,
#     **DB_PARAMS
# )


@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    except psycopg2.DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            connection_pool.putconn(conn)


def execute_query(query, params=None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Failed to execute query: {e}")


def fetch_results(query, params=None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query, params)
                return cur.fetchall()
            except Exception as e:
                print(f"Failed to fetch results: {e}")
                return []


def execute_transaction(queries):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                for query, params in queries:
                    cur.execute(query, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Transaction failed: {e}")


def close_connection_pool():
    connection_pool.closeall()
