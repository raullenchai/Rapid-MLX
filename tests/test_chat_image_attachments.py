"""Image upload validation and durable attachment contracts."""

import io
import sqlite3
import struct
import zlib

import httpx
import pytest
from PIL import Image

from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.file_extraction import DocumentError, extract_upload
from rapid_mlx.chat_runtime.store import Store


def fixture_image(fmt, size=(2, 3)):
    output = io.BytesIO()
    Image.new("RGB", size).save(output, format=fmt)
    return output.getvalue()


@pytest.mark.parametrize(
    ("fmt", "name", "mime"),
    [
        ("PNG", "x.png", "image/png"),
        ("JPEG", "x.jpg", "image/jpeg"),
        ("WEBP", "x.webp", "image/webp"),
        ("GIF", "x.gif", "image/gif"),
    ],
)
async def test_upload_and_binary_integrity(tmp_path, fmt, name, mime):
    data = fixture_image(fmt)
    assert extract_upload(data, name) == {
        "kind": "image",
        "mime": mime,
        "width": 2,
        "height": 3,
    }
    store = await Store.open(tmp_path)
    try:
        conversation = await store.create_conversation()
        other = await store.create_conversation()
        app = create_app(store, None, "secret")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            response = await client.post(
                PREFIX + f"/conversations/{conversation['id']}/attachments",
                params={"name": name},
                content=data,
            )
        assert response.status_code == 201, response.text
        attachment = response.json()
        assert attachment["mime"] == mime
        assert await store.read_attachment_bytes(
            conversation["id"], attachment["id"]
        ) == (mime, data)
        with pytest.raises(KeyError):
            await store.read_attachment_bytes(other["id"], attachment["id"])
        with pytest.raises(ValueError, match="cannot be read as text"):
            await store.read_attachment(conversation["id"], attachment["id"])
        (store.blob_dir / attachment["digest"]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="integrity"):
            await store.read_attachment_bytes(conversation["id"], attachment["id"])
    finally:
        await store.close()


async def test_invalid_and_oversize_upload(tmp_path):
    store = await Store.open(tmp_path)
    try:
        c = await store.create_conversation()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(store, None, "secret")),
            base_url="http://test",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            url = PREFIX + f"/conversations/{c['id']}/attachments"
            for data, name, status, code in [
                (fixture_image("PNG"), "x.jpg", 422, "image_invalid"),
                (b"bad", "x.png", 422, "image_invalid"),
                (fixture_image("PNG", (8193, 1)), "x.png", 413, "image_too_large"),
            ]:
                response = await client.post(url, params={"name": name}, content=data)
                assert response.status_code == status
                assert response.json()["detail"]["code"] == code
        with pytest.raises(DocumentError, match="20 MiB"):
            extract_upload(b"x" * (20 * 1024 * 1024 + 1), "x.png")
        with pytest.raises(ValueError, match="invalid image"):
            await store.put_attachment(c["id"], "x", b"x", "image/unknown")
    finally:
        await store.close()


async def test_migrate_existing_attachment(tmp_path):
    store = await Store.open(tmp_path)
    c = await store.create_conversation()
    attachment = await store.put_attachment(c["id"], "old.txt", b"hello")
    await store.close()
    db = sqlite3.connect(tmp_path / "history.sqlite")
    db.execute("ALTER TABLE attachments DROP COLUMN mime")
    db.execute("PRAGMA user_version=4")
    db.commit()
    db.close()
    store = await Store.open(tmp_path)
    try:
        assert await store.read_attachment_bytes(c["id"], attachment["id"]) == (
            "text/plain",
            b"hello",
        )
        assert await store.read_attachment(c["id"], attachment["id"]) == "hello"
    finally:
        await store.close()


def test_extraction_validation_paths():
    assert extract_upload(b"hello", "note.txt") == {"text": "hello"}
    for data, name, code in [
        (fixture_image("PNG"), "wrong.jpg", "image_invalid"),
        (b"bad", "bad.png", "image_invalid"),
        (fixture_image("PNG", (8193, 1)), "large.png", "image_too_large"),
    ]:
        with pytest.raises(DocumentError) as error:
            extract_upload(data, name)
        assert error.value.code == code
    data = bytearray(fixture_image("PNG"))
    data[16:24] = struct.pack(">II", 20000, 20000)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    with pytest.raises(DocumentError) as error:
        extract_upload(bytes(data), "bomb.png")
    assert error.value.code == "image_too_large"
