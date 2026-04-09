import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from novel_kb.utils.logger import logger


class AuthDatabase:
    """Manage kb_auth.db SQLite database for user authentication and KB ownership."""

    DEFAULT_DB_PATH = Path.home() / ".novel_knowledge_base" / "kb_auth.db"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._ensure_db_dir()
        self._init_db()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database with required tables."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    api_key TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS kb_ownership (
                    name TEXT NOT NULL,
                    novel_id TEXT NOT NULL,
                    PRIMARY KEY (name, novel_id)
                );
            """)
            conn.commit()
            logger.info("Auth database initialized at: %s", self.db_path)

    def create_user(self, name: str, api_key_hash: str) -> str:
        """Create a new user and return the user_id."""
        user_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (user_id, name, api_key) VALUES (?, ?, ?)",
                (user_id, name, api_key_hash),
            )
            conn.commit()
            logger.info("User created: name=%s, user_id=%s", name, user_id)
            return user_id

    def delete_user(self, name: str) -> bool:
        """Delete a user and all their KB ownership records. Returns True if deleted."""
        with self._get_connection() as conn:
            # First delete kb_ownership records
            cursor = conn.execute(
                "DELETE FROM kb_ownership WHERE name = ?", (name,)
            )
            conn.commit()
            kb_deleted = cursor.rowcount

            # Then delete the user
            cursor = conn.execute(
                "DELETE FROM users WHERE name = ?", (name,)
            )
            conn.commit()
            user_deleted = cursor.rowcount

            if user_deleted > 0:
                logger.info("User deleted: name=%s, kb_records_deleted=%d", name, kb_deleted)
                return True
            return False

    def get_user_by_name(self, name: str) -> Optional[dict]:
        """Get user by name. Returns dict with user_id, name, api_key, created_at or None."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT user_id, name, api_key, created_at FROM users WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def assign_kb(self, name: str, novel_id: str) -> bool:
        """Assign a KB file to a user. Returns True if created, False if already exists."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO kb_ownership (name, novel_id) VALUES (?, ?)",
                    (name, novel_id),
                )
                conn.commit()
                logger.info("KB assigned: name=%s, novel_id=%s", name, novel_id)
                return True
        except sqlite3.IntegrityError:
            # Already exists
            return False

    def unassign_kb(self, name: str, novel_id: str) -> bool:
        """Remove KB assignment from a user. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM kb_ownership WHERE name = ? AND novel_id = ?",
                (name, novel_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_user_kb(self, name: str) -> list[str]:
        """List all KB novel_ids accessible by a user."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT novel_id FROM kb_ownership WHERE name = ? ORDER BY novel_id",
                (name,),
            )
            return [row["novel_id"] for row in cursor.fetchall()]

    def user_exists(self, name: str) -> bool:
        """Check if a user exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM users WHERE name = ?", (name,)
            )
            return cursor.fetchone() is not None