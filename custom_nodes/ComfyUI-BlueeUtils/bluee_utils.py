# BlueeUtils: utilitários para deletar arquivo e editar JSON
import json
import os
import tempfile
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse
import uuid


class BlueeUtils:
    def __init__(self):
        pass

    # ================== Implementações ==================

    def _do_delete(self, path: str):
        if not path or not path.strip():
            return ("Caminho não informado", "{}", 400)
        path = path.strip()
        if not os.path.isabs(path):
            return ("O caminho deve ser absoluto", "{}", 400)
        if not os.path.exists(path):
            return (f"Arquivo não encontrado: {path}", "{}", 404)
        if not os.path.isfile(path):
            return ("O caminho informado não é um arquivo", "{}", 400)
        try:
            os.remove(path)
            return (f"Removido: {path}", "{}", 200)
        except Exception as e:
            return (f"Falha ao remover arquivo: {str(e)}", "{}", 500)

    def _do_edit(self, source: str, expression: str, string: str = ""):
        try:
            data = json.loads(source) if (source and source.strip()) else None
        except json.JSONDecodeError as e:
            return (f"JSON de entrada inválido: {str(e)}", "{}", 400)

        safe_builtins = {
            "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
            "list": list, "dict": dict, "any": any, "all": all, "str": str,
            "int": int, "float": float, "bool": bool, "range": range,
            "enumerate": enumerate, "zip": zip,
        }
        safe_globals = {"__builtins__": safe_builtins}
        local_vars = {"data": data, "string": string}
        try:
            value: Any = eval(expression, safe_globals, local_vars)
        except Exception as e:
            return (f"Erro ao avaliar expressão: {str(e)}", "{}", 400)

        try:
            if isinstance(value, str):
                return (value, "{}", 200)
            return (json.dumps(value, ensure_ascii=False), "{}", 200)
        except Exception:
            return (str(value), "{}", 200)

    # ================== Helpers para Edit com template n8n ==================
    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        path = path.strip()
        # Remove prefixo de root (ex.: $json, $input1)
        if path.startswith("$json"):
            path = path[5:]
        elif path.startswith("$input1"):
            path = path[7:]
        elif path.startswith("$input2"):
            path = path[7:]
        elif path.startswith("$input3"):
            path = path[7:]
        elif path.startswith("$input4"):
            path = path[7:]
        elif path.startswith("$input5"):
            path = path[7:]
        elif path.startswith("$input6"):
            path = path[7:]
        # Tokeniza dot/brackets
        tokens = []
        buf = ""
        i = 0
        while i < len(path):
            ch = path[i]
            if ch == '.':
                if buf:
                    tokens.append(buf)
                    buf = ""
                i += 1
            elif ch == '[':
                if buf:
                    tokens.append(buf)
                    buf = ""
                j = i + 1
                depth = 1
                while j < len(path) and depth > 0:
                    if path[j] == '[':
                        depth += 1
                    elif path[j] == ']':
                        depth -= 1
                    j += 1
                inner = path[i+1:j-1].strip()
                if inner.startswith("'") or inner.startswith('"'):
                    key = inner.strip("'\"")
                else:
                    try:
                        key = int(inner)
                    except ValueError:
                        key = inner
                tokens.append(key)
                i = j
            else:
                buf += ch
                i += 1
        if buf:
            tokens.append(buf)

        cur = root
        for t in tokens:
            if t == "":
                continue
            if isinstance(t, int):
                if isinstance(cur, list) and 0 <= t < len(cur):
                    cur = cur[t]
                else:
                    raise KeyError(f"Índice inválido: {t}")
            else:
                if isinstance(cur, dict) and t in cur:
                    cur = cur[t]
                else:
                    raise KeyError(f"Chave não encontrada: {t}")
        return cur

    @classmethod
    def _apply_template(cls, data: Any, template_str: str, context: dict | None = None) -> Any:
        # Tenta parsear o template como JSON
        try:
            template = json.loads(template_str) if (template_str and template_str.strip()) else None
        except json.JSONDecodeError as e:
            # Se não for JSON válido, trata como string pura com placeholders
            return cls._replace_placeholders_in_string(data, template_str, context)

        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, str):
                return cls._replace_placeholders_in_string(data, node, context)
            return node

        return walk(template)

    @classmethod
    def _replace_placeholders_in_string(cls, data: Any, s: str, context: dict | None = None):
        import re
        pattern_full = re.compile(r"^\s*\{\{\s*(.*?)\s*\}\}\s*$")
        pattern_any = re.compile(r"\{\{\s*(.*?)\s*\}\}")

        m = pattern_full.match(s)
        if m:
            expr = m.group(1)
            if expr.startswith("$json"):
                return cls._resolve_path(data, expr)
            if context is not None and any(expr.startswith(k) for k in ("$input1","$input2","$input3","$input4","$input5","$input6")):
                root = context.get(expr.split('.',1)[0].split('[',1)[0], None)
                return cls._resolve_path(root, expr)
            # Se não começa com $json, retorna string original
            return s

        def repl(match):
            expr = match.group(1)
            try:
                if expr.startswith("$json"):
                    val = cls._resolve_path(data, expr)
                    return str(val)
                if context is not None and any(expr.startswith(k) for k in ("$input1","$input2","$input3","$input4","$input5","$input6")):
                    root = context.get(expr.split('.',1)[0].split('[',1)[0], None)
                    val = cls._resolve_path(root, expr)
                    return str(val)
            except Exception:
                return ""
            return match.group(0)

        return pattern_any.sub(repl, s)


class BlueeUtilsDelete(BlueeUtils):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"forceInput": True}),
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Aceita qualquer tipo ligado ao trigger sem validação estrita
        return True

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def run(self, trigger, path: str):
        # Executa deleção e não retorna nada
        try:
            # Apenas executa se algo estiver conectado ao trigger
            if trigger is not None:
                self._do_delete(path)
        except Exception:
            # Silencia erros para manter sem saída; logs podem ser adicionados se necessário
            pass
        return ()


# BlueeUtilsEdit removido por solicitação


class blueeInput:
    """
    Node simples que recebe um texto e o retorna.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def output(self, text: str):
        if text and "{{uuid}}" in text:
            # Substitui todas as ocorrências por UUIDs distintos
            while "{{uuid}}" in text:
                text = text.replace("{{uuid}}", str(uuid.uuid4()), 1)
        return (text,)


class BlueeTemplateString:
    """
    Substitui o placeholder {{input}} dentro de um template de texto pelo valor de entrada.
    Ex.: input="nome" e template="{{input}}-audio.mp4" => "nome-audio.mp4".
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "", "multiline": False}),
                "template": ("STRING", {"default": "{{input}}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, input: str, template: str):
        if template is None:
            template = ""
        value = input if input is not None else ""
        # substitui todas ocorrências
        result = template.replace("{{input}}", value)
        return (result,)

class BlueeUtilsDownloadTemp:
    """
    Baixa um arquivo de uma URL, salva em arquivo temporário e retorna o caminho completo.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "filename": ("STRING", {"default": "", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, url: str, filename: str = "", path: str = ""):
        if not url or not url.strip():
            return ("",)
        url = url.strip()

        parsed = urlparse(url)
        base_name = os.path.basename(parsed.path)
        suffix = ""
        if base_name and "." in base_name:
            suffix = os.path.splitext(base_name)[1]

        # Mapeia alguns MIME comuns para extensões preferidas
        mime_ext_overrides = {
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/aac": ".aac",
            "audio/ogg": ".ogg",
            "audio/opus": ".opus",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/flac": ".flac",
            "audio/x-flac": ".flac",
            "audio/webm": ".webm",
        }

        try:
            # Define diretório de destino
            if path and path.strip():
                target_dir = path.strip()
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception:
                    return ("",)
            else:
                target_dir = tempfile.gettempdir()

            if filename and filename.strip():
                # Se filename não possui extensão, tenta acrescentar
                name_only = filename.strip()
                name_ext = os.path.splitext(name_only)[1]
                if not name_ext and suffix:
                    name_only = name_only + suffix
                tmp_path = os.path.join(target_dir, name_only)
            else:
                # Se não foi passado filename, tenta usar o basename da URL ou cria temporário no diretório alvo
                if base_name:
                    name_only = base_name
                    tmp_path = os.path.join(target_dir, name_only)
                else:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=target_dir)
                    tmp_path = tmp_file.name
                    tmp_file.close()
        except Exception:
            return ("",)

        try:
            try:
                import requests
                with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                    r.raise_for_status()
                    # Se filename não tinha extensão nem URL path forneceu, tenta por Content-Type
                    current_ext = os.path.splitext(tmp_path)[1]
                    if not current_ext:
                        # Tenta Content-Disposition filename
                        cd = r.headers.get("Content-Disposition", "")
                        if "filename=" in cd:
                            import re
                            m = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)|filename=\"?([^\";]+)\"?", cd)
                            if m:
                                dispo_name = m.group(1) or m.group(2)
                                if dispo_name:
                                    ext_cd = os.path.splitext(dispo_name)[1]
                                    if ext_cd:
                                        tmp_path = tmp_path + ext_cd
                                        current_ext = ext_cd
                        if not current_ext:
                            import mimetypes
                            ct = r.headers.get("Content-Type", "").split(";")[0].strip()
                            guessed = mime_ext_overrides.get(ct)
                            if not guessed and ct:
                                guessed = mimetypes.guess_extension(ct)
                            if guessed:
                                tmp_path = tmp_path + guessed
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                # Verifica arquivo final
                try:
                    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                        return ("",)
                except Exception:
                    return ("",)
            except Exception:
                import urllib.request
                with urllib.request.urlopen(url, timeout=60) as resp:
                    # Se filename não tinha extensão nem URL path forneceu, tenta por Content-Type
                    current_ext = os.path.splitext(tmp_path)[1]
                    if not current_ext:
                        import mimetypes
                        try:
                            ct = resp.info().get_content_type()
                        except Exception:
                            ct = ""
                        guessed = mime_ext_overrides.get(ct)
                        if not guessed and ct:
                            guessed = mimetypes.guess_extension(ct)
                        if guessed:
                            tmp_path = tmp_path + guessed
                    with open(tmp_path, "wb") as out:
                        out.write(resp.read())
                try:
                    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                        return ("",)
                except Exception:
                    return ("",)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return ("",)

        return (tmp_path,)


class BlueeUtilsReadTempImage:
    """
    Lê uma imagem de um caminho temporário e retorna como IMAGE (tensor [1,H,W,C] em 0..1).
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, path: str):
        if not path or not os.path.isfile(path):
            raise Exception("Imagem não encontrada")
        try:
            from PIL import Image
            import numpy as np
            import torch
        except Exception:
            raise Exception("Dependências de imagem indisponíveis (PIL/numpy/torch)")
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        h, w, c = arr.shape
        tensor = torch.from_numpy(arr).reshape(1, h, w, c)
        return (tensor,)


class BlueeUtilsReadTempAudio:
    """
    Lê um arquivo de áudio e retorna como AUDIO {"waveform": tensor[B,C,T], "sample_rate": int}.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, path: str):
        if not path or not os.path.isfile(path):
            raise Exception("Áudio não encontrado")
        # 1) Tenta com torchaudio
        try:
            import torchaudio
            import torch
            waveform, sample_rate = torchaudio.load(path)  # [C, T]
            waveform = waveform.unsqueeze(0)  # [1, C, T]
            return ({"waveform": waveform, "sample_rate": int(sample_rate)},)
        except Exception:
            pass

        # 2) Tenta com PyAV (suporta MP3/MP4/OGG/etc.)
        try:
            import av
            import numpy as np
            import torch
            container = av.open(path)
            audio_stream = None
            for s in container.streams:
                if s.type == 'audio':
                    audio_stream = s
                    break
            if audio_stream is None:
                container.close()
                raise Exception("Sem stream de áudio")
            sample_rate = audio_stream.rate or 44100
            chunks = []
            for frame in container.decode(audio=audio_stream):
                arr = frame.to_ndarray()  # [C, S]
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32) / (32768.0 if arr.dtype == np.int16 else 1.0)
                chunks.append(arr)
            container.close()
            if not chunks:
                raise Exception("Sem dados de áudio decodificados")
            data = np.concatenate(chunks, axis=1)  # [C, T]
            waveform = torch.from_numpy(data).unsqueeze(0)  # [1, C, T]
            return ({"waveform": waveform, "sample_rate": int(sample_rate)},)
        except Exception:
            pass

        # 3) Fallback para WAV PCM simples
        try:
            ext = os.path.splitext(path)[1].lower()
            temp_wav = None
            # Se não for WAV/WAVE, tenta converter com ffmpeg
            if ext not in (".wav", ".wave"):
                if shutil.which("ffmpeg") is None:
                    raise Exception("Formato não suportado sem torchaudio/av")
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_wav = tmp_file.name
                tmp_file.close()
                try:
                    # Converte para WAV PCM s16le, 44.1kHz, preservando canais
                    subprocess.run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", path,
                        "-ar", "44100",
                        temp_wav
                    ], check=True)
                    path_to_read = temp_wav
                except Exception as conv_e:
                    # Limpa e propaga erro
                    try:
                        os.remove(temp_wav)
                    except Exception:
                        pass
                    raise Exception(f"Falha na conversão ffmpeg: {str(conv_e)}")
            else:
                path_to_read = path
            import wave
            import numpy as np
            import torch
            with wave.open(path_to_read, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(n_frames)
            if sampwidth == 2:
                dtype = np.int16
                scale = 32768.0
            else:
                dtype = np.uint8
                scale = 255.0
            data = np.frombuffer(frames, dtype=dtype)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).T  # [C, T]
            else:
                data = data.reshape(1, -1)
            waveform = torch.from_numpy(data.astype(np.float32) / scale).unsqueeze(0)  # [1, C, T]
            return ({"waveform": waveform, "sample_rate": int(sample_rate)},)
        except Exception as e:
            raise Exception(f"Falha ao ler áudio: {str(e)}")
        finally:
            # Remove WAV temporário se criado
            try:
                if 'temp_wav' in locals() and temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception:
                pass
class BlueeFileExists:
    """
    Verifica se um arquivo existe no caminho informado.
    Retorna duas saídas booleanas: true e false.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {}),
            }
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("true", "false")
    FUNCTION = "run"
    CATEGORY = "utils"
    OUTPUT_NODE = False

    def run(self, value):
        # Se for string, checa se é um arquivo; se for boolean, usa o próprio valor; caso contrário, usa truthiness
        if isinstance(value, str):
            exists = bool(value) and os.path.isfile(value)
            out_val = value
        elif isinstance(value, bool):
            exists = value
            out_val = "true" if value else "false"
        else:
            exists = bool(value)
            out_val = str(value)
        return (out_val, not exists)


class BlueeCombineImagesAudioH264:
    """
    Concatena imagens (frames) + áudio opcional em vídeo MP4 (H.264, alta qualidade).

    - frame_rate: taxa de quadros (padrão 29)
    - filename_prefix: prefixo do arquivo
    - save_output: se true, salva em output; senão, em temp
    - crf: fator de qualidade do x264 (menor = melhor; 18 é visualmente lossless na maioria dos casos)
    - preset: preset do x264 (trade-off velocidade/qualidade)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "frame_rate": ("INT", {"default": 29, "min": 1, "max": 240, "step": 1}),
                "filename_prefix": ("STRING", {"default": "BlueeVideo", "multiline": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "preset": ([
                    "ultrafast", "superfast", "veryfast", "faster", "fast",
                    "medium", "slow", "slower", "veryslow"
                ], {"default": "slow"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(self, images, frame_rate, filename_prefix, save_output, crf, preset, audio=None):
        try:
            import numpy as np
            import torch
            from PIL import Image  # noqa: F401 (apenas para garantir dependências comuns presentes)
            import folder_paths
        except Exception as e:
            raise Exception(f"Dependências ausentes: {str(e)}")

        if images is None:
            return ("",)

        # Suporta batch ou lista de frames
        if isinstance(images, dict) and 'samples' in images:
            frames = images['samples']
        else:
            frames = images

        if isinstance(frames, torch.Tensor) and frames.size(0) == 0:
            return ("",)

        if isinstance(frames, torch.Tensor):
            # forma esperada [N,H,W,C] com valores 0..1
            if len(frames.shape) == 4:
                pass
            elif len(frames.shape) == 5:
                # remove batch extra se vier como [B,N,H,W,C]
                frames = frames[0]
            else:
                raise Exception("Formato de imagens não suportado")
            num_frames = frames.size(0)
            height = int(frames.size(1))
            width = int(frames.size(2))
            # converte para uint8 RGB
            frames_bytes_iter = ( ( (frames[i].clamp(0,1) * 255.0).to(torch.uint8).cpu().numpy() ).tobytes() for i in range(num_frames) )
        else:
            # lista de arrays/tensors
            seq = list(frames)
            if len(seq) == 0:
                return ("",)
            sample = seq[0]
            if isinstance(sample, torch.Tensor):
                height = int(sample.size(-3))
                width = int(sample.size(-2))
                def tobytes(t):
                    return (t.clamp(0,1) * 255.0).to(torch.uint8).cpu().numpy().tobytes()
                frames_bytes_iter = ( tobytes(x) for x in seq )
            else:
                arr = np.array(sample)
                height, width = int(arr.shape[0]), int(arr.shape[1])
                frames_bytes_iter = ( np.array(x).astype(np.uint8).tobytes() for x in seq )

        # Define diretório/arquivo de saída
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)

        # Arquivos alvo
        base_mp4 = os.path.join(full_output_folder, f"{filename}.mp4")
        final_mp4 = base_mp4
        # Usamos um arquivo temporário para o vídeo sem áudio e depois substituímos pelo final
        tmp_video_only = os.path.join(full_output_folder, f"{filename}.tmpvideo.mp4")

        # Garante diretório
        os.makedirs(full_output_folder, exist_ok=True)

        # Passo 1: gerar vídeo H.264 a partir dos framess
        def _get_ffmpeg_path():
            # 1) Variáveis de ambiente
            for k in ("BLUEE_FFMPEG", "FFMPEG", "FFMPEG_BINARY"):
                p = os.environ.get(k)
                if p and os.path.isfile(p):
                    return p
                if p and shutil.which(p):
                    return shutil.which(p)
            # 2) PATH do sistema
            p = shutil.which("ffmpeg")
            if p:
                return p
            # 3) imageio-ffmpeg (binário empacotado via pip)
            try:
                import imageio_ffmpeg
                p = imageio_ffmpeg.get_ffmpeg_exe()
                if p and os.path.isfile(p):
                    return p
            except Exception:
                pass
            return None

        ffmpeg = _get_ffmpeg_path()
        if not ffmpeg:
            raise Exception("ffmpeg não encontrado. Instale com: apt-get install -y ffmpeg ou pip install imageio-ffmpeg")
        args = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(frame_rate),
            "-i", "-",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-preset", str(preset),
            "-crf", str(crf),
            "-movflags", "+faststart",
            tmp_video_only,
        ]

        try:
            with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
                for fb in frames_bytes_iter:
                    proc.stdin.write(fb)
                proc.stdin.flush()
                proc.stdin.close()
                proc.wait()
                if proc.returncode != 0:
                    raise Exception("FFmpeg falhou ao criar o vídeo")
        except Exception as e:
            raise Exception(f"Falha ao gerar vídeo: {str(e)}")

        # Passo 2 (opcional): mux com áudio
        if audio is not None and isinstance(audio, dict) and 'waveform' in audio and 'sample_rate' in audio:
            try:
                import torch
                waveform = audio['waveform']  # [B,C,T]
                sample_rate = int(audio['sample_rate'])
                if not isinstance(waveform, torch.Tensor):
                    raise Exception("waveform inválido")
                channels = int(waveform.size(1))
                audio_bytes = waveform.squeeze(0).transpose(0,1).contiguous().cpu().numpy().astype('float32').tobytes()

                # Escreve diretamente no arquivo final base_mp4
                output_with_audio = base_mp4
                mux_args = [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_video_only,
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-f", "f32le", "-i", "-",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    output_with_audio,
                ]
                res = subprocess.run(mux_args, input=audio_bytes, capture_output=True)
                if res.returncode != 0:
                    raise Exception(res.stderr.decode("utf-8", errors="ignore"))
                final_mp4 = output_with_audio
                # Remove o vídeo temporário sem áudio
                try:
                    if os.path.exists(tmp_video_only):
                        os.remove(tmp_video_only)
                except Exception:
                    pass
            except Exception as e:
                # Mantém o vídeo sem áudio, mas informa erro
                raise Exception(f"Falha ao muxar áudio: {str(e)}")
        else:
            # Sem áudio: renomeia o arquivo temporário para o final
            try:
                if os.path.exists(final_mp4):
                    os.remove(final_mp4)
            except Exception:
                pass
            try:
                os.replace(tmp_video_only, final_mp4)
            except Exception:
                # Se falhar, ao menos tenta manter o temporário como resultado
                final_mp4 = tmp_video_only

        return (final_mp4,)
NODE_CLASS_MAPPINGS = {
    "BlueeUtilsDelete": BlueeUtilsDelete,
    "blueeInput": blueeInput,
    "BlueeTemplateString": BlueeTemplateString,
    "BlueeUtilsDownloadTemp": BlueeUtilsDownloadTemp,
    "BlueeUtilsReadTempImage": BlueeUtilsReadTempImage,
    "BlueeUtilsReadTempAudio": BlueeUtilsReadTempAudio,
    "BlueeFileExists": BlueeFileExists,
    "BlueeCombineImagesAudioH264": BlueeCombineImagesAudioH264,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueeUtilsDelete": "Bluee Utils - Delete File",
    "blueeInput": "blueeInput",
    "BlueeTemplateString": "Bluee Template String",
    "BlueeUtilsDownloadTemp": "Bluee Utils - Download Temp",
    "BlueeUtilsReadTempImage": "Bluee Utils - Read Temp Image",
    "BlueeUtilsReadTempAudio": "Bluee Utils - Read Temp Audio",
    "BlueeFileExists": "bluee - file exists",
    "BlueeCombineImagesAudioH264": "Bluee - Combine Images + Audio (H.264 MP4)",
}


