import runpod
import traceback

INIT_ERROR = None


def load_model():
    import os
    import shutil

    print("[init] Verifying FaceFusion installation...", flush=True)

    if not os.path.isfile('/facefusion/facefusion.py'):
        raise RuntimeError('/facefusion/facefusion.py not found')

    for tool in ['ffmpeg', 'curl']:
        if not shutil.which(tool):
            raise RuntimeError(f'{tool} not found in PATH')

    models_dir = '/facefusion/.assets/models'
    if not os.path.isdir(models_dir):
        raise RuntimeError(f'Models directory not found: {models_dir}')

    required_models = [
        'ghost_1_256.onnx',
        'gfpgan_1.4.onnx',
        'retinaface_10g.onnx',
        '2dfan4.onnx',
        'arcface_w600k_r50.onnx',
    ]
    missing = [m for m in required_models if not os.path.isfile(os.path.join(models_dir, m))]
    if missing:
        raise RuntimeError(f'Missing model files: {", ".join(missing)}')

    print("[init] FaceFusion ready.", flush=True)


try:
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def handler(job):
    if INIT_ERROR:
        return {'error': f'Model failed to load:\n{INIT_ERROR}'}

    import base64
    import os
    import shutil
    import subprocess
    import tempfile
    import uuid

    inp = job['input']

    video_b64 = inp.get('video_base64')
    if not video_b64:
        return {'error': 'video_base64 is required'}

    source_face_b64 = inp.get('source_face_base64')
    if not source_face_b64:
        return {'error': 'source_face_base64 is required'}

    face_swap_model = inp.get('face_swap_model', 'ghost_1_256')
    face_enhancer_model = inp.get('face_enhancer_model', 'gfpgan_1.4')
    reference_face_position = int(inp.get('reference_face_position', 0))
    reference_face_distance = float(inp.get('reference_face_distance', 0.6))
    face_selector_mode = inp.get('face_selector_mode', 'reference')
    reference_frame_number = int(inp.get('reference_frame_number', 0))
    output_video_quality = int(inp.get('output_video_quality', 80))

    source_path = None
    target_path = None
    output_path = None
    job_dir = None

    try:
        tmp_source = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_source.write(base64.b64decode(source_face_b64))
        tmp_source.close()
        source_path = tmp_source.name

        tmp_target = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_target.write(base64.b64decode(video_b64))
        tmp_target.close()
        target_path = tmp_target.name

        tmp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_output.close()
        output_path = tmp_output.name

        job_id = str(uuid.uuid4())
        job_dir = f'/tmp/facefusion-jobs/{job_id}'
        os.makedirs(job_dir, exist_ok=True)

        processors = ['face_swapper']
        if face_enhancer_model:
            processors.append('face_enhancer')

        cmd = [
            'python', '/facefusion/facefusion.py', 'headless-run',
            '-s', source_path,
            '-t', target_path,
            '-o', output_path,
            '--processors', *processors,
            '--face-swapper-model', face_swap_model,
            '--face-detector-model', 'retinaface',
            '--face-selector-mode', face_selector_mode,
            '--reference-face-position', str(reference_face_position),
            '--reference-face-distance', str(reference_face_distance),
            '--reference-frame-number', str(reference_frame_number),
            '--output-video-quality', str(output_video_quality),
            '--output-video-encoder', 'libx264',
            '--output-video-preset', 'medium',
            '--execution-providers', 'cuda',
            '--temp-path', '/tmp/facefusion',
            '--jobs-path', job_dir,
        ]

        if face_enhancer_model:
            cmd.extend(['--face-enhancer-model', face_enhancer_model])

        print(f"[handler] Running: {' '.join(cmd)}", flush=True)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd='/facefusion',
        )

        if result.returncode == 3:
            return {'error': 'Content flagged by NSFW filter'}

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else 'no stderr'
            return {'error': f'FaceFusion failed (exit {result.returncode}): {stderr_tail}'}

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else 'no stderr'
            return {'error': f'Output file missing or empty. stderr: {stderr_tail}'}

        file_size = os.path.getsize(output_path)
        with open(output_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        b64_len = len(video_base64)
        print(f"[handler] Output file: {file_size / 1024 / 1024:.2f} MB, base64 length: {b64_len}, estimated payload: {b64_len / 1024 / 1024:.2f} MB", flush=True)

        return {
            'video_base64': video_base64,
            'format': 'mp4',
        }

    except subprocess.TimeoutExpired:
        return {'error': 'FaceFusion timed out after 600 seconds'}
    except Exception as e:
        return {'error': str(e)}
    finally:
        for path in [source_path, target_path, output_path]:
            if path and os.path.isfile(path):
                os.unlink(path)
        if job_dir and os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)


runpod.serverless.start({'handler': handler})
