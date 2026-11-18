import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.v1.export import router
from app.database import Base, get_db
from app.models import Project, QAPair


@pytest.fixture
def export_db(tmp_path):
    # File-backed SQLite is shared across the request worker and the test thread.
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def test_export_api_source_split_and_missing_provenance(export_db, tmp_path):
    db_session = export_db
    db_session.add(Project(id="project", name="Test", domain="Test"))
    db_session.flush()
    for i in range(10):
        db_session.add(QAPair(
            id=f"row-{i}", project_id="project", question=f"Question {i}?", answer="Answer",
            source_type="rag_ollama", metadata_json={"rag": True, "retrieved_context": {
                "passages": [{"source_id": f"source-{i // 2}"}],
            }},
        ))
    db_session.commit()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db_session
    with patch("app.api.v1.export.get_settings", return_value=SimpleNamespace(output_path=tmp_path)):
        with TestClient(app) as client:
            url = "/api/v1/projects/project/export"
            response = client.post(url, json={"format": "json", "split_strategy": "source",
                                              "train_split": 0.6, "val_split": 0.2, "test_split": 0.2})
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["split_counts"] == {"train": 6, "val": 2, "test": 2}
            source_locations = {}
            for split, filename in payload["files"].items():
                rows = json.loads((tmp_path / "project" / filename).read_text())
                for row in rows:
                    source_id = row["retrieved_context"]["passages"][0]["source_id"]
                    assert source_locations.setdefault(source_id, split) == split
            bad = client.post(url, json={"split_strategy": "source", "train_split": 0.8,
                                         "val_split": 0.2, "test_split": 0.2})
            assert bad.status_code == 422
            assert client.post(url, json={"split_strategy": "unknown"}).status_code == 422
            record = db_session.get(QAPair, "row-0")
            record.metadata_json = None
            db_session.commit()
            assert client.post(url, json={"split_strategy": "source"}).status_code == 422
            assert client.post(url, json={"format": "json"}).status_code == 200
