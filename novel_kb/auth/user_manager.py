import hashlib
from typing import Optional

from novel_kb.auth.database import AuthDatabase
from novel_kb.utils.logger import logger


class UserManager:
    """Manage users with SHA256 hashed API keys."""

    def __init__(self, db_path: Optional[str] = None):
        from pathlib import Path
        self.db = AuthDatabase(db_path=Path(db_path) if db_path else None)

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key using SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def register_user(self, name: str, api_key: str) -> str:
        """
        Register a new user.
        Returns user_id if successful.
        Raises ValueError if user already exists.
        """
        if self.db.user_exists(name):
            raise ValueError(f"User already exists: {name}")

        api_key_hash = self.hash_api_key(api_key)
        user_id = self.db.create_user(name, api_key_hash)
        logger.info("User registered: name=%s", name)
        return user_id

    def delete_user(self, name: str) -> bool:
        """
        Delete a user and all their KB ownership records.
        Returns True if deleted, False if user not found.
        """
        return self.db.delete_user(name)

    def assign_kb(self, name: str, novel_id: str) -> bool:
        """
        Assign a KB file to a user.
        Returns True if assigned, False if already assigned.
        Raises ValueError if user not found.
        """
        if not self.db.user_exists(name):
            raise ValueError(f"User not found: {name}")

        return self.db.assign_kb(name, novel_id)

    def list_user_kb(self, name: str) -> list[str]:
        """
        List all KB novel_ids accessible by a user.
        Returns empty list if user not found.
        """
        return self.db.list_user_kb(name)

    def verify_user(self, name: str, api_key: str) -> bool:
        """
        Verify user credentials.
        Returns True if valid, False otherwise.
        """
        user = self.db.get_user_by_name(name)
        if not user:
            return False
        api_key_hash = self.hash_api_key(api_key)
        return user["api_key"] == api_key_hash