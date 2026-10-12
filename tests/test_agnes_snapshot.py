from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mirai.core.moe.artifacts.downloads as downloads
from mirai.core.moe.artifacts.downloads import (
    MoEArtifactDownloadSpec,
    PinnedArtifactFile,
    artifact_allow_patterns,
    get_moe_artifact_download_spec,
    get_moe_artifact_manifest_metadata,
    verify_artifact_manifest_source,
    verify_pinned_artifact_files,
)
from mirai.core.moe.artifacts.manifest import write_download_manifest
from scripts.download import (
    _download_gitee_mirror,
    _git_blob_sha1,
    _request_with_retries,
    _snapshot_file_inventory,
)


AGNES_VARIANT = "agnes-video-2.0-moe"
AGNES_REVISION = "fc3971e378b53c766baa06898bfe84d7a56653ea"


def test_agnes_release_declares_all_pinned_split_checkpoint_parts() -> None:
    spec = get_moe_artifact_download_spec(AGNES_VARIANT)

    assert spec.repo_id == "Agnes-AI/Agnes-Video-2.0-MoE"
    assert spec.revision == AGNES_REVISION
    assert spec.artifact_license == "apache-2.0"
    assert [item.path for item in spec.expected_files] == [
        f"checkpoint_parts/ckpt_best.pt.part-{number:03d}" for number in range(7)
    ]
    assert [item.size for item in spec.expected_files] == [
        *(8 * 1024**3 for _ in range(6)),
        5_933_844_437,
    ]
    assert all(len(item.sha256) == 64 for item in spec.expected_files)


def test_agnes_download_patterns_keep_weights_required_and_source_optional() -> None:
    weights_only = artifact_allow_patterns(AGNES_VARIANT)
    with_source = artifact_allow_patterns(AGNES_VARIANT, include_source_code=True)

    assert weights_only == tuple(
        f"checkpoint_parts/ckpt_best.pt.part-{number:03d}" for number in range(7)
    )
    assert with_source is not None
    assert "inference_code/**" in with_source
    assert set(weights_only or ()).issubset(with_source)


def test_agnes_auxiliary_defaults_fetch_only_runtime_components_and_allow_override() -> None:
    assert artifact_allow_patterns("agnes-wan2.2-ti2v-5b") == (
        "Wan2.2_VAE.pth",
        "LICENSE*",
    )
    assert artifact_allow_patterns("agnes-omnivae") == (
        "models/vae/audio_only/recon/*",
        "LICENSE*",
    )
    assert artifact_allow_patterns(
        "agnes-omnivae", ("models/vae/audio_only/config.yaml",)
    ) == ("models/vae/audio_only/config.yaml",)


def test_agnes_auxiliary_sources_are_independent_and_license_explicit() -> None:
    metadata = get_moe_artifact_manifest_metadata(AGNES_VARIANT)
    sources = {item["role"]: item for item in metadata["auxiliary_sources"]}

    assert metadata["revision"] == AGNES_REVISION
    assert sources["text_encoder"] == {
        "role": "text_encoder",
        "repo_id": "google/t5gemma-9b-9b-ul2",
        "revision": "7071a2ace9c8353e12e723e5382cc7256ee41d9f",
        "artifact_license": "gemma",
        "access": "public_mirror",
        "download_url": "https://gitee.com/hf-models/t5gemma-9b-9b-ul2.git",
        "download_revision": "7071a2ace9c8353e12e723e5382cc7256ee41d9f",
        "required_patterns": [
            "README.md",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00009.safetensors",
            "model-00002-of-00009.safetensors",
            "model-00003-of-00009.safetensors",
            "model-00004-of-00009.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
        "license_url": "https://ai.google.dev/gemma/terms",
    }
    assert sources["video_vae"]["repo_id"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert sources["video_vae"]["artifact_license"] == "apache-2.0"
    assert sources["audio_vae"]["repo_id"] == "OpenMOSS-Team/OmniVAE"
    assert sources["audio_vae"]["artifact_license"] == "apache-2.0"
    assert get_moe_artifact_download_spec("agnes-t5gemma-9b-9b-ul2").revision == (
        sources["text_encoder"]["revision"]
    )
    assert get_moe_artifact_download_spec("agnes-wan2.2-ti2v-5b").revision == (
        sources["video_vae"]["revision"]
    )
    assert get_moe_artifact_download_spec("agnes-omnivae").revision == (
        sources["audio_vae"]["revision"]
    )


def test_generic_pinned_verifier_checks_size_and_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "part.bin"
    payload.write_bytes(b"agnes")
    expected = PinnedArtifactFile(
        path="part.bin",
        size=5,
        sha256=hashlib.sha256(b"agnes").hexdigest(),
    )
    synthetic = MoEArtifactDownloadSpec(
        variant="synthetic-pinned",
        repo_id="example/synthetic",
        spec_id="synthetic",
        expected_files=(expected,),
    )
    monkeypatch.setattr(
        downloads,
        "MOE_ARTIFACT_DOWNLOAD_SPECS",
        downloads.MOE_ARTIFACT_DOWNLOAD_SPECS + (synthetic,),
    )

    verify_pinned_artifact_files(tmp_path, "synthetic-pinned")
    payload.write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_pinned_artifact_files(tmp_path, "synthetic-pinned")


def test_download_inventory_records_pinned_hash_without_rehashing(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_parts" / "ckpt_best.pt.part-000"
    path.parent.mkdir()
    path.write_bytes(b"part")
    digest = hashlib.sha256(b"part").hexdigest()

    assert _snapshot_file_inventory(
        tmp_path,
        expected_hashes={"checkpoint_parts/ckpt_best.pt.part-000": digest},
    ) == [
        {
            "path": "checkpoint_parts/ckpt_best.pt.part-000",
            "size": 4,
            "status": "downloaded",
            "sha256": digest,
        }
    ]


def test_registered_auxiliary_manifest_requires_pinned_source_revision(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    write_download_manifest(
        tmp_path,
        {
            "status": "downloaded",
            "variant": "agnes-omnivae",
            "repo_id": "OpenMOSS-Team/OmniVAE",
            "revision": "wrong",
            "files": [{"path": "config.json", "size": 2}],
        },
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        verify_artifact_manifest_source(tmp_path, "agnes-omnivae")


def test_public_git_lfs_mirror_download_is_integrity_checked_and_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shard_bytes = b"synthetic-safetensors"
    oid = hashlib.sha256(shard_bytes).hexdigest()
    source_index = json.dumps(
        {
            "metadata": {"total_size": len(shard_bytes)},
            "weight_map": {
                "model.encoder.layer.weight": "model-00001-of-00002.safetensors",
                "model.decoder.layer.weight": "model-00002-of-00002.safetensors",
            },
        }
    ).encode()
    config = json.dumps(
        {
            "architectures": ["T5GemmaForConditionalGeneration"],
            "is_encoder_decoder": True,
            "encoder": {},
        }
    ).encode()
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\nsize {len(shard_bytes)}\n"
    ).encode()
    blobs = {_git_blob_sha1(value): value for value in (config, source_index, pointer)}
    blob_ids = list(blobs)
    signed_url = "https://signed.invalid/private-query-value"

    class Response:
        def __init__(self, *, payload=None, content=b""):
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Session:
        def get(self, url, **kwargs):
            if "/git/trees/" in url:
                return Response(
                    payload={
                        "tree": [
                            {"path": "config.json", "type": "blob", "sha": blob_ids[0]},
                            {
                                "path": "model.safetensors.index.json",
                                "type": "blob",
                                "sha": blob_ids[1],
                            },
                            {
                                "path": "model-00001-of-00002.safetensors",
                                "type": "blob",
                                "sha": blob_ids[2],
                            },
                        ]
                    }
                )
            if "/git/blobs/" in url:
                content = blobs[url.rsplit("/", 1)[-1]]
                return Response(payload={"content": base64.b64encode(content).decode()})
            assert url == signed_url
            return Response(content=shard_bytes)

        def post(self, url, **kwargs):
            return Response(
                payload={"objects": [{"actions": {"download": {"href": signed_url}}}]}
            )

    monkeypatch.setitem(__import__("sys").modules, "requests", SimpleNamespace(Session=Session))
    artifact = MoEArtifactDownloadSpec(
        variant="synthetic-mirror",
        repo_id="original/source",
        spec_id="synthetic",
        revision="original-revision",
        required_patterns=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ),
        mirror_url="https://gitee.com/example/mirror.git",
        mirror_revision="mirror-revision",
        index_key_prefixes=("model.encoder.",),
        encoder_only_config=True,
    )
    output = tmp_path / "snapshot"

    _download_gitee_mirror(artifact, output)

    derived = json.loads((output / "model.safetensors.index.json").read_text())
    retained = json.loads((output / "model.safetensors.index.source.json").read_text())
    derived_config = json.loads((output / "config.json").read_text())
    assert set(derived["weight_map"]) == {"model.encoder.layer.weight"}
    assert len(retained["weight_map"]) == 2
    assert derived["metadata"]["source_repo_id"] == "original/source"
    assert derived_config["architectures"] == ["T5GemmaEncoderModel"]
    assert derived_config["is_encoder_decoder"] is False
    assert (output / "model-00001-of-00002.safetensors").read_bytes() == shard_bytes
    assert "private-query-value" not in capsys.readouterr().out


def test_mirror_request_retries_transient_status_and_sanitizes_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HttpFailure(Exception):
        def __init__(self, status: int):
            self.response = SimpleNamespace(status_code=status)

    class Response:
        def __init__(self, status: int):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HttpFailure(self.status_code)

    class Session:
        calls = 0

        def get(self, url, **kwargs):
            self.calls += 1
            return Response(502 if self.calls < 3 else 200)

    monkeypatch.setattr("scripts.download.time.sleep", lambda _: None)
    session = Session()
    assert _request_with_retries(
        session, "get", "https://signed.invalid/object?secret=value"
    ).status_code == 200
    assert session.calls == 3

    with pytest.raises(RuntimeError) as failure:
        _request_with_retries(
            SimpleNamespace(get=lambda *args, **kwargs: Response(502)),
            "get",
            "https://signed.invalid/object?secret=value",
            attempts=1,
        )
    assert "secret=value" not in str(failure.value)


def test_mirror_stream_resumes_partial_payload_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_bytes = b"first-second"
    oid = hashlib.sha256(shard_bytes).hexdigest()
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\nsize {len(shard_bytes)}\n"
    ).encode()
    pointer_sha = _git_blob_sha1(pointer)
    signed_url = "https://signed.invalid/object?credential=private"

    class Response:
        def __init__(self, *, payload=None, chunks=(), status_code=200):
            self._payload = payload
            self.chunks = chunks
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for item in self.chunks:
                if isinstance(item, Exception):
                    raise item
                yield item

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Session:
        payload_calls = 0

        def get(self, url, **kwargs):
            if "/git/trees/" in url:
                return Response(payload={"tree": [{
                    "path": "model.safetensors", "type": "blob", "sha": pointer_sha
                }]})
            if "/git/blobs/" in url:
                return Response(payload={"content": base64.b64encode(pointer).decode()})
            self.payload_calls += 1
            if self.payload_calls == 1:
                return Response(chunks=(b"first-", OSError("connection reset")))
            assert kwargs["headers"] == {"Range": "bytes=6-"}
            return Response(chunks=(b"second",), status_code=206)

        def post(self, url, **kwargs):
            return Response(payload={"objects": [{"actions": {"download": {"href": signed_url}}}]})

    session = Session()
    monkeypatch.setitem(__import__("sys").modules, "requests", SimpleNamespace(Session=lambda: session))
    monkeypatch.setattr("scripts.download.time.sleep", lambda _: None)
    artifact = MoEArtifactDownloadSpec(
        variant="synthetic-resume",
        repo_id="original/source",
        spec_id="synthetic",
        required_patterns=("model.safetensors",),
        mirror_url="https://gitee.com/example/mirror.git",
        mirror_revision="revision",
    )

    output = tmp_path / "snapshot"
    _download_gitee_mirror(artifact, output)

    assert (output / "model.safetensors").read_bytes() == shard_bytes
    assert session.payload_calls == 2
