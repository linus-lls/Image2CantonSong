from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Optional, List
from schemas import LyricsPromptBundle, SongResult

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

YUE_OFFICIAL_PYTHON = "/userhome/cs5/u3665806/anaconda3/envs/yue_official/bin/python"
YUE_OFFICIAL_REPO = Path("/userhome/cs5/u3665806/YuE")
YUE_OFFICIAL_INFER_DIR = YUE_OFFICIAL_REPO / "inference"

def _find_audio_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

def _preferred_audio(folder: Path) -> Optional[Path]:
    files = _find_audio_files(folder)
    if not files:
        return None

    def norm(p: Path) -> str:
        return str(p).replace("\\", "/")

    mix_candidates = [p for p in files if "vocoder/mix" in norm(p) and p.name.endswith("_mixed.mp3")]
    if mix_candidates:
        return max(mix_candidates, key=lambda p: p.stat().st_mtime)

    recons_mix_candidates = [p for p in files if "recons/mix" in norm(p) and p.name.endswith("_mixed.mp3")]
    if recons_mix_candidates:
        return max(recons_mix_candidates, key=lambda p: p.stat().st_mtime)

    return max(files, key=lambda p: p.stat().st_mtime)

def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _write_meta(meta_path: Path, meta: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_song_auto(
    bundle: LyricsPromptBundle,
    out_dir: str = "outputs",
    stage1_model: str = "m-a-p/YuE-s1-7B-anneal-zh-icl",
    stage2_model: str = "m-a-p/YuE-s2-1B-general",
    run_n_segments: int = 2,
    stage2_batch_size: int = 1,
    max_new_tokens: int = 1200,
    repetition_penalty: float = 1.1,
    use_single_track_icl: bool = False,
    use_dual_track_icl: bool = False,
    prompt_audio_path: str = "",
    vocal_prompt_path: str = "",
    instrumental_prompt_path: str = "",
    prompt_start_time: int = 0,
    prompt_end_time: int = 30,
    extra_cli_args: str = "",
) -> SongResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "song_metadata.json"
    yue_input = out / "yue_inputs"
    yue_output = out / "yue_outputs"
    yue_input.mkdir(parents=True, exist_ok=True)
    yue_output.mkdir(parents=True, exist_ok=True)

    genre_txt = yue_input / "genre.txt"
    lyrics_txt = yue_input / "lyrics.txt"

    genre_text = bundle.genre_prompt.strip() or "cantopop ballad piano guitar sentimental female airy vocal Cantonese"
    # lyrics_text = bundle.lyrics_text.strip()
    lyrics_text = bundle.lyrics_text

    _write_text(genre_txt, genre_text)
    _write_text(lyrics_txt, lyrics_text)

    cmd = [
        YUE_OFFICIAL_PYTHON,
        "infer.py",
        "--cuda_idx", "0",
        "--stage1_model", stage1_model,
        "--stage2_model", stage2_model,
        "--genre_txt", str(genre_txt.resolve()),
        "--lyrics_txt", str(lyrics_txt.resolve()),
        "--run_n_segments", str(run_n_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", str(yue_output.resolve()),
        "--max_new_tokens", str(max_new_tokens),
        "--repetition_penalty", str(repetition_penalty),
    ]

    if use_dual_track_icl and vocal_prompt_path and instrumental_prompt_path:
        cmd.extend([
            "--use_dual_tracks_prompt",
            "--vocal_track_prompt_path", str(Path(vocal_prompt_path).resolve()),
            "--instrumental_track_prompt_path", str(Path(instrumental_prompt_path).resolve()),
            "--prompt_start_time", str(prompt_start_time),
            "--prompt_end_time", str(prompt_end_time),
        ])
    elif use_single_track_icl and prompt_audio_path:
        cmd.extend([
            "--use_audio_prompt",
            "--audio_prompt_path", str(Path(prompt_audio_path).resolve()),
            "--prompt_start_time", str(prompt_start_time),
            "--prompt_end_time", str(prompt_end_time),
        ])

    if extra_cli_args.strip():
        cmd.extend(extra_cli_args.strip().split())

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(YUE_OFFICIAL_REPO.resolve()),
        str((YUE_OFFICIAL_REPO / "inference").resolve()),
        str((YUE_OFFICIAL_REPO / "inference" / "xcodec_mini_infer").resolve()),
        env.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    proc = subprocess.run(
        cmd,
        cwd=str(YUE_OFFICIAL_INFER_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    audio = _preferred_audio(yue_output)

    meta = {
        "backend": "project_demo_with_official_yue_subprocess",
        "success": bool(audio is not None),
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "run_n_segments": run_n_segments,
        "stage2_batch_size": stage2_batch_size,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "visual_anchor": bundle.raw_meta.get("visual_anchor", ""),
        "llm_backend": bundle.raw_meta.get("llm_backend", ""),
        "python_bin": YUE_OFFICIAL_PYTHON,
        "repo": str(YUE_OFFICIAL_REPO),
        "cwd": str(YUE_OFFICIAL_INFER_DIR),
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
        "found_audio": str(audio) if audio else "",
    }

    _write_meta(meta_path, meta)

    if audio is None:
        raise RuntimeError(
            f"YuE finished without a discoverable audio file.\\n\\n"
            f"Metadata path: {meta_path}\\n\\n"
            f"STDOUT tail:\\n{meta['stdout_tail']}\\n\\n"
            f"STDERR tail:\\n{meta['stderr_tail']}"
        )

    if proc.returncode != 0:
        meta["warning"] = "Audio file exists, but infer.py exited non-zero during later steps."
        meta["success"] = True
        _write_meta(meta_path, meta)

    return SongResult(
        success=True,
        final_song_path=str(audio.resolve()),
        metadata_path=str(meta_path),
        generation_log=meta,
    )
