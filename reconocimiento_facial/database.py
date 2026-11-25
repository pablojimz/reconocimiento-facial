"""Local identity storage with envelope encryption."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from base64 import b64decode, b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fcntl import LOCK_EX, flock
from pathlib import Path
from threading import Lock

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .paths import USER_DATABASE

DEFAULT_PASSWORD = "biometria-demo"
PASSWORD_ENV = "BIOMETRIC_DB_PASSWORD"

_MAGIC = b"MAEBDB\x00\x01"
_SALT_SIZE = 16
_NONCE_SIZE = 12
_KEY_SIZE = 32
_TAG_SIZE = 16
_HEADER_SIZE = len(_MAGIC) + _SALT_SIZE + _NONCE_SIZE + _KEY_SIZE + _TAG_SIZE
_CATALOG = b"catalog"
_USER = b"user/"


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    status: str
    source_image: str
    template: tuple[float, ...] = ()


@dataclass(frozen=True)
class _UnlockedKey:
    header: bytes
    master_key: bytes


def _password_key(password: str, salt: bytes) -> bytes:
    if not password.strip():
        raise ValueError("The database password must not be empty.")
    return Scrypt(salt=salt, length=_KEY_SIZE, n=2**17, r=8, p=1).derive(
        password.encode("utf-8")
    )


def _derive_key(master_key: bytes, purpose: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_SIZE,
        salt=None,
        info=_MAGIC + purpose,
    ).derive(master_key)


def _seal(key: bytes, plaintext: bytes, context: bytes) -> bytes:
    nonce = os.urandom(_NONCE_SIZE)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, context)


def _unseal(key: bytes, encrypted: bytes, context: bytes) -> bytes:
    nonce = encrypted[:_NONCE_SIZE]
    return AESGCM(key).decrypt(nonce, encrypted[_NONCE_SIZE:], context)


def _wrap_key(master_key: bytes, password: str) -> _UnlockedKey:
    salt = os.urandom(_SALT_SIZE)
    prefix = _MAGIC + salt
    wrapped = _seal(_password_key(password, salt), master_key, prefix)
    return _UnlockedKey(prefix + wrapped, master_key)


class UserDatabase:
    """Read and update an encrypted identity catalog."""

    fieldnames = ("user_id", "status", "source_image", "template")

    def __init__(
        self, path: str | Path = USER_DATABASE, *, password: str | None = None
    ) -> None:
        self.path = Path(path)
        self._password = (
            password
            if password is not None
            else os.getenv(PASSWORD_ENV, DEFAULT_PASSWORD)
        )
        self._unlocked: _UnlockedKey | None = None
        self._mutex = Lock()

    def list_users(self) -> list[UserRecord]:
        with self._mutex:
            users, _ = self._read_users()
        return users

    def add_user(
        self,
        user_id: str,
        source_image: str = "",
        *,
        template: np.ndarray | None = None,
        replace_existing: bool = False,
    ) -> UserRecord:
        normalized_id = user_id.strip()
        if not normalized_id:
            raise ValueError("The user identifier must not be empty.")
        with self._write_lock():
            users, key = self._read_users()
            remaining = [
                user for user in users if user.user_id != normalized_id
            ]
            if len(remaining) != len(users) and not replace_existing:
                raise ValueError(f"User '{normalized_id}' already exists.")
            features = (
                self._validate_template(template)
                if template is not None
                else ()
            )
            status = "enrolled" if features else "pending_template"
            record = UserRecord(normalized_id, status, source_image, features)
            self._write([*remaining, record], key)
        return record

    def remove_user(self, user_id: str) -> bool:
        with self._write_lock():
            users, key = self._read_users()
            remaining = [user for user in users if user.user_id != user_id]
            if len(users) == len(remaining):
                return False
            self._write(remaining, key)
        return True

    def migrate_csv(self) -> bool:
        """Encrypt a plaintext CSV in place and report whether it changed."""
        with self._write_lock():
            contents = self.path.read_bytes()
            if contents.startswith(_MAGIC):
                self._read_users(contents)
                return False
            try:
                users = self._read_legacy_csv(contents)
            except csv.Error as error:
                raise ValueError(
                    f"Invalid CSV at {self.path}: {error}"
                ) from error
            self._write(users)
        return True

    def change_password(self, new_password: str) -> None:
        """Rewrap the master key with the new password."""
        with self._write_lock():
            try:
                contents = self.path.read_bytes()
            except FileNotFoundError:
                key = _wrap_key(
                    AESGCM.generate_key(bit_length=256), new_password
                )
                self._write([], key)
            else:
                old_key, catalog = self._unlock(contents)
                self._open_catalog(old_key, catalog)
                key = _wrap_key(old_key.master_key, new_password)
                self._replace(key.header + catalog)
            self._password = new_password
            self._unlocked = key

    def _unlock(self, contents: bytes) -> tuple[_UnlockedKey, bytes]:
        if not contents.startswith(_MAGIC):
            if contents.startswith(b"user_id,status,source_image"):
                raise ValueError("Run 'database migrate' to encrypt the CSV.")
            raise ValueError("Unknown encrypted database format.")
        if len(contents) < _HEADER_SIZE + _NONCE_SIZE + _TAG_SIZE:
            raise ValueError("The encrypted database file is incomplete.")

        header, catalog = contents[:_HEADER_SIZE], contents[_HEADER_SIZE:]
        cached = self._unlocked
        if cached is not None and cached.header == header:
            return cached, catalog
        prefix_end = len(_MAGIC) + _SALT_SIZE
        prefix = header[:prefix_end]
        salt = prefix[len(_MAGIC) :]
        try:
            master_key = _unseal(
                _password_key(self._password, salt),
                header[prefix_end:],
                prefix,
            )
        except InvalidTag as error:
            raise ValueError(
                "Incorrect password or damaged database header."
            ) from error
        key = _UnlockedKey(header, master_key)
        self._unlocked = key
        return key, catalog

    @staticmethod
    def _open_catalog(key: _UnlockedKey, catalog: bytes) -> bytes:
        try:
            return _unseal(
                _derive_key(key.master_key, _CATALOG),
                catalog,
                _MAGIC + _CATALOG,
            )
        except InvalidTag as error:
            raise ValueError(
                "The encrypted database was damaged or modified."
            ) from error

    def _read_users(
        self, contents: bytes | None = None
    ) -> tuple[list[UserRecord], _UnlockedKey | None]:
        if contents is None:
            try:
                contents = self.path.read_bytes()
            except FileNotFoundError:
                return [], None
        key, catalog = self._unlock(contents)
        users = []
        for user_id, encoded in json.loads(self._open_catalog(key, catalog)):
            purpose = _USER + user_id.encode("utf-8")
            try:
                plaintext = _unseal(
                    _derive_key(key.master_key, purpose),
                    b64decode(encoded),
                    _MAGIC + purpose,
                )
            except InvalidTag as error:
                raise ValueError(
                    "An encrypted database record is damaged."
                ) from error
            status, source_image, template = json.loads(plaintext)
            users.append(
                UserRecord(user_id, status, source_image, tuple(template))
            )
        return users, key

    def _read_legacy_csv(self, contents: bytes) -> list[UserRecord]:
        reader = csv.DictReader(
            io.StringIO(contents.decode("utf-8"), newline=""), strict=True
        )
        valid_headers = [list(self.fieldnames), list(self.fieldnames[:-1])]
        if reader.fieldnames not in valid_headers:
            raise ValueError(
                f"Invalid CSV columns at {self.path}: {self.fieldnames}"
            )
        users = []
        user_ids = set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Invalid CSV row at line {reader.line_num}.")
            user_id = row["user_id"]
            if not user_id.strip() or user_id in user_ids:
                raise ValueError(
                    f"Empty or duplicate user ID at line {reader.line_num}."
                )
            user_ids.add(user_id)
            encoded = row.pop("template", "")
            template = (
                self._validate_template(json.loads(encoded)) if encoded else ()
            )
            users.append(UserRecord(**row, template=template))
        return users

    def _write(
        self, users: list[UserRecord], key: _UnlockedKey | None = None
    ) -> None:
        if key is None:
            key = _wrap_key(
                AESGCM.generate_key(bit_length=256), self._password
            )
        rows = []
        for user in users:
            purpose = _USER + user.user_id.encode("utf-8")
            plaintext = json.dumps(
                (user.status, user.source_image, user.template),
                allow_nan=False,
            ).encode("utf-8")
            encrypted = _seal(
                _derive_key(key.master_key, purpose),
                plaintext,
                _MAGIC + purpose,
            )
            rows.append((user.user_id, b64encode(encrypted).decode("ascii")))
        catalog = _seal(
            _derive_key(key.master_key, _CATALOG),
            json.dumps(rows).encode("utf-8"),
            _MAGIC + _CATALOG,
        )
        self._replace(key.header + catalog)
        self._unlocked = key

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        path = self.path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lock a separate file because writes replace the database inode.
        lock_path = path.with_name(path.name + ".lock")
        with self._mutex, lock_path.open("a") as lock_file:
            flock(lock_file, LOCK_EX)
            yield

    def _replace(self, contents: bytes) -> None:
        path = self.path.resolve()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(contents)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_template(template: object) -> tuple[float, ...]:
        values = np.asarray(template, dtype=np.float32)
        if values.shape != (512,) or not np.isfinite(values).all():
            raise ValueError("The template must contain 512 finite values.")
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-5):
            raise ValueError("The template must have unit L2 norm.")
        return tuple(float(value) for value in values)
