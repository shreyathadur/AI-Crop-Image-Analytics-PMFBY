"""Small persistence adapter: MongoDB when configured, otherwise SQLite."""
import json
import sqlite3
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.config import DATABASE_URL, MONGODB_DATABASE, MONGODB_URI, ROOT


def _sqlite_path():
    value = DATABASE_URL.removeprefix("sqlite:///")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _mongo():
    if not MONGODB_URI:
        return None
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
    client.admin.command("ping")
    return client[MONGODB_DATABASE]


def initialize():
    db = _mongo()
    if db is not None:
        db.users.create_index("email", unique=True)
        db.analyses.create_index([("user_id", 1), ("created_at", -1)])
        db.fields.create_index([("user_id", 1), ("is_active", 1)])
        db.analyses.create_index([("user_id", 1), ("field_id", 1), ("created_at", -1)])
        db.weather_cache.create_index("cache_key", unique=True)
        db.password_reset_tokens.create_index("token_hash", unique=True)
        db.password_reset_tokens.create_index([("user_id", 1), ("used", 1)])
        return
    with sqlite3.connect(_sqlite_path()) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS analyses_user_created ON analyses(user_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS password_reset_tokens (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, expires_at TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS reset_tokens_user ON password_reset_tokens(user_id, used);
        CREATE TABLE IF NOT EXISTS fields (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, field_name TEXT NOT NULL, crop_name TEXT NOT NULL, variety TEXT, state TEXT, district TEXT, village TEXT, latitude REAL, longitude REAL, area REAL, area_unit TEXT, sowing_date TEXT, notes TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1);
        CREATE INDEX IF NOT EXISTS fields_user_active ON fields(user_id, is_active);
        CREATE TABLE IF NOT EXISTS weather_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL);
        """)
        analysis_columns = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        if "field_id" not in analysis_columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN field_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS analyses_user_field_created ON analyses(user_id, field_id, created_at DESC)")
    if db is not None:
        db.password_reset_tokens.create_index("token_hash", unique=True)
        db.password_reset_tokens.create_index([("user_id", 1), ("used", 1)])


def create_user(email, password_hash):
    user_id, now = str(uuid4()), datetime.now(timezone.utc).isoformat()
    db = _mongo()
    if db is not None:
        db.users.insert_one({"_id": user_id, "email": email, "password_hash": password_hash, "created_at": now})
    else:
        with sqlite3.connect(_sqlite_path()) as conn:
            conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (user_id, email, password_hash, now))
    return {"id": user_id, "email": email}


def get_user_by_email(email):
    db = _mongo()
    if db is not None:
        row = db.users.find_one({"email": email})
        return {"id": row["_id"], "email": row["email"], "password_hash": row["password_hash"]} if row else None
    with sqlite3.connect(_sqlite_path()) as conn:
        row = conn.execute("SELECT id,email,password_hash FROM users WHERE email=?", (email,)).fetchone()
    return {"id": row[0], "email": row[1], "password_hash": row[2]} if row else None


def update_password(user_id, password_hash):
    db = _mongo()
    if db is not None:
        return db.users.update_one({"_id": user_id}, {"$set": {"password_hash": password_hash}}).matched_count == 1
    with sqlite3.connect(_sqlite_path()) as conn:
        cur = conn.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, user_id))
        return cur.rowcount == 1


FIELD_COLUMNS = ("field_name", "crop_name", "variety", "state", "district", "village",
                 "latitude", "longitude", "area", "area_unit", "sowing_date", "notes")


def _field_document(row):
    if not row:
        return None
    return {"id": row.get("_id", row.get("id")), **{key: row.get(key) for key in FIELD_COLUMNS},
            "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
            "is_active": bool(row.get("is_active", True))}


def create_field(user_id, values):
    field_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    doc = {"_id": field_id, "user_id": user_id, **{key: values.get(key) for key in FIELD_COLUMNS},
           "created_at": now, "updated_at": now, "is_active": True}
    db = _mongo()
    if db is not None:
        db.fields.insert_one(doc)
    else:
        with sqlite3.connect(_sqlite_path()) as conn:
            cols = ("id", "user_id", *FIELD_COLUMNS, "created_at", "updated_at", "is_active")
            conn.execute(f"INSERT INTO fields ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                         tuple(field_id if key == "id" else user_id if key == "user_id" else now if key in {"created_at", "updated_at"} else 1 if key == "is_active" else values.get(key) for key in cols))
    return _field_document(doc)


def list_fields(user_id, include_archived=False):
    db = _mongo()
    if db is not None:
        query = {"user_id": user_id}
        if not include_archived:
            query["is_active"] = True
        return [_field_document(row) for row in db.fields.find(query).sort("created_at", -1)]
    with sqlite3.connect(_sqlite_path()) as conn:
        query = "SELECT id,user_id," + ",".join(FIELD_COLUMNS) + ",created_at,updated_at,is_active FROM fields WHERE user_id=?"
        params = [user_id]
        if not include_archived:
            query += " AND is_active=1"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
    names = ("id", "user_id", *FIELD_COLUMNS, "created_at", "updated_at", "is_active")
    return [_field_document(dict(zip(names, row))) for row in rows]


def get_field(field_id, user_id, include_archived=False):
    db = _mongo()
    if db is not None:
        query = {"_id": field_id, "user_id": user_id}
        if not include_archived:
            query["is_active"] = True
        return _field_document(db.fields.find_one(query))
    with sqlite3.connect(_sqlite_path()) as conn:
        query = "SELECT id,user_id," + ",".join(FIELD_COLUMNS) + ",created_at,updated_at,is_active FROM fields WHERE id=? AND user_id=?"
        params = [field_id, user_id]
        if not include_archived:
            query += " AND is_active=1"
        row = conn.execute(query, params).fetchone()
    names = ("id", "user_id", *FIELD_COLUMNS, "created_at", "updated_at", "is_active")
    return _field_document(dict(zip(names, row))) if row else None


def update_field(field_id, user_id, values):
    now = datetime.now(timezone.utc).isoformat()
    db = _mongo()
    updates = {key: values.get(key) for key in FIELD_COLUMNS} | {"updated_at": now}
    if db is not None:
        result = db.fields.update_one({"_id": field_id, "user_id": user_id, "is_active": True}, {"$set": updates})
        return get_field(field_id, user_id) if result.matched_count else None
    with sqlite3.connect(_sqlite_path()) as conn:
        cur = conn.execute("UPDATE fields SET " + ",".join(f"{key}=?" for key in (*FIELD_COLUMNS, "updated_at")) + " WHERE id=? AND user_id=? AND is_active=1",
                           tuple(values.get(key) for key in FIELD_COLUMNS) + (now, field_id, user_id))
        if not cur.rowcount:
            return None
    return get_field(field_id, user_id)


def archive_field(field_id, user_id):
    now = datetime.now(timezone.utc).isoformat()
    db = _mongo()
    if db is not None:
        result = db.fields.update_one({"_id": field_id, "user_id": user_id, "is_active": True}, {"$set": {"is_active": False, "updated_at": now}})
        return result.modified_count == 1
    with sqlite3.connect(_sqlite_path()) as conn:
        cur = conn.execute("UPDATE fields SET is_active=0,updated_at=? WHERE id=? AND user_id=? AND is_active=1", (now, field_id, user_id))
        return cur.rowcount == 1


def get_cached_weather(cache_key, max_age_seconds):
    db = _mongo()
    if db is not None:
        row = db.weather_cache.find_one({"cache_key": cache_key})
        if not row:
            return None
        payload, fetched_at = row.get("payload"), row.get("fetched_at")
    else:
        with sqlite3.connect(_sqlite_path()) as conn:
            row = conn.execute("SELECT payload,fetched_at FROM weather_cache WHERE cache_key=?", (cache_key,)).fetchone()
        if not row:
            return None
        payload, fetched_at = row
        payload = json.loads(payload)
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds()
    except (TypeError, ValueError):
        return None
    return payload if 0 <= age < max_age_seconds else None


def save_weather_cache(cache_key, payload):
    fetched_at = datetime.now(timezone.utc).isoformat()
    db = _mongo()
    if db is not None:
        db.weather_cache.update_one({"cache_key": cache_key},
                                    {"$set": {"payload": payload, "fetched_at": fetched_at}}, upsert=True)
        return
    with sqlite3.connect(_sqlite_path()) as conn:
        conn.execute("INSERT INTO weather_cache (cache_key,payload,fetched_at) VALUES (?,?,?) ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,fetched_at=excluded.fetched_at",
                     (cache_key, json.dumps(payload), fetched_at))


def create_password_reset(user_id, token_hash, expires_at):
    reset_id, now = str(uuid4()), datetime.now(timezone.utc).isoformat()
    db = _mongo()
    if db is not None:
        db.password_reset_tokens.update_many({"user_id": user_id, "used": False}, {"$set": {"used": True}})
        db.password_reset_tokens.insert_one({"_id": reset_id, "user_id": user_id, "token_hash": token_hash, "expires_at": expires_at, "used": False, "created_at": now})
        return reset_id
    with sqlite3.connect(_sqlite_path()) as conn:
        conn.execute("UPDATE password_reset_tokens SET used=1 WHERE user_id=? AND used=0", (user_id,))
        conn.execute("INSERT INTO password_reset_tokens VALUES (?, ?, ?, ?, 0, ?)", (reset_id, user_id, token_hash, expires_at, now))
    return reset_id


def consume_password_reset(token_hash, now):
    db = _mongo()
    if db is not None:
        row = db.password_reset_tokens.find_one({"token_hash": token_hash, "used": False, "expires_at": {"$gt": now}})
        if not row:
            return None
        user = db.users.find_one({"_id": row["user_id"]})
        if not user:
            return None
        result = db.password_reset_tokens.update_one({"_id": row["_id"], "used": False}, {"$set": {"used": True}})
        return row["user_id"] if result.modified_count else None
    with sqlite3.connect(_sqlite_path()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT user_id FROM password_reset_tokens WHERE token_hash=? AND used=0 AND expires_at>?", (token_hash, now)).fetchone()
        if not row or not conn.execute("SELECT 1 FROM users WHERE id=?", (row[0],)).fetchone():
            return None
        conn.execute("UPDATE password_reset_tokens SET used=1 WHERE user_id=? AND used=0", (row[0],))
        return row[0]


def save_analysis(payload):
    item = dict(payload); item.setdefault("id", str(uuid4()))
    item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    db = _mongo()
    if db is not None:
        db.analyses.insert_one({"_id": item["id"], **item})
    else:
        with sqlite3.connect(_sqlite_path()) as conn:
            conn.execute("INSERT INTO analyses (id,user_id,payload,created_at,field_id) VALUES (?, ?, ?, ?, ?)",
                         (item["id"], item["user_id"], json.dumps(item), item["created_at"], item.get("field_id")))
    return item


def get_analysis(analysis_id, user_id):
    db = _mongo()
    if db is not None:
        row = db.analyses.find_one({"_id": analysis_id, "user_id": user_id})
        return {k: v for k, v in row.items() if k != "_id"} if row else None
    with sqlite3.connect(_sqlite_path()) as conn:
        row = conn.execute("SELECT payload FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id)).fetchone()
    return json.loads(row[0]) if row else None


def list_analyses(user_id):
    db = _mongo()
    if db is not None:
        return [{k: v for k, v in row.items() if k != "_id"} for row in db.analyses.find({"user_id": user_id}).sort("created_at", -1)]
    with sqlite3.connect(_sqlite_path()) as conn:
        rows = conn.execute("SELECT payload FROM analyses WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def delete_analysis(analysis_id, user_id):
    db = _mongo()
    if db is not None:
        return db.analyses.delete_one({"_id": analysis_id, "user_id": user_id}).deleted_count > 0
    with sqlite3.connect(_sqlite_path()) as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id))
        return cur.rowcount > 0
