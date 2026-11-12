import asyncio
import base64
import json
import logging
import os

# Localizar as bibliotecas do MSYS2 no Windows.
# Manter a referencia para que o diretorio continue registrado.
_msys2_dlls = None
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _msys2_path = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(_msys2_path):
        _msys2_dlls = os.add_dll_directory(_msys2_path)
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    APP_HOST,
    APP_PORT,
    BASE_DIR,
    SED_LOGIN_URL,
    SED_VALIDA_URL,
    SED_SUBSCRIPTION_KEY,
    IPTV_TOKEN_URL,
    IPTV_BASE_URL,
)
from client import HttpCloakClient, _get_browser_context, _generate_traceparent
from captcha_solver import tms_apply_with_captcha
from database import init_db, get_cached_leiasp_quiz_answer, save_cached_leiasp_quiz_answer
from ai_solver import resolve_task_answers, resolve_leiasp_objective, resolve_leiasp_dissertative
from matific_client import (
    MatificClient,
    translate_matific_slug,
    _aggregate_played_episodes
)
from leiasp_client import LeiaSPClient, LeiaSPAuthError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("saladopassado.app")

_sessions: dict[str, dict] = {}
_SESSION_TTL = 24 * 3600
# Per-user task detail cache: (username, task_id) -> task_data.
# Keying only by task_id leaked one student's apply payload
# (answer id, attempt state) into another student's session.
_task_cache: dict[tuple[str, int], dict] = {}
_delayed_jobs: dict[str, dict] = {}

_matific_session_cache: dict[str, dict] = {}
_MATIFIC_SESSION_TTL = 20 * 60
_MATIFIC_EPISODES_TTL = 5 * 60
_MATIFIC_STATE_TTL = 3 * 60
_matific_locks: dict[str, asyncio.Lock] = {}
_matific_jobs: dict[str, dict] = {}
_matific_batches: dict[str, dict] = {}

_task_batches: dict[str, dict] = {}
# Shared by batches and individual requests in this server process. Only task
# preparation (opening/CAPTCHA/AI) takes a slot; delays and answer sends do not.
_task_preparation_semaphore = asyncio.Semaphore(5)
_leiasp_jobs: dict[str, dict] = {}
_leiasp_locks: dict[str, asyncio.Lock] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized successfully.")
    yield

app = FastAPI(title="Sala do Passado — SalaDoPassado", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

async def get_current_user(request: Request) -> dict:
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _sessions:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:].strip()
            
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=401, detail="Sessão expirada ou não autenticado")
    user = _sessions[session_id]
    if time.time() - user.get("created_at", time.time()) > _SESSION_TTL:
        _sessions.pop(session_id, None)
        raise HTTPException(status_code=401, detail="Sessão expirada ou não autenticado")
    return user


def _tms_targets_query(rooms: list, nick: str) -> str:
    """Build publication_target query fragment from /room/user payload.

    Mirrors the official web client (see harfiles/tarefas_update.har):
    room names, room:nick variants and numeric group_categories ids.
    """
    targets: list[str] = []
    for r in rooms or []:
        rname = r.get("name")
        if rname:
            targets.append(f"publication_target={rname}")
            if nick:
                targets.append(f"publication_target={rname}:{nick}")
        for cat in r.get("group_categories", []) or []:
            cid = cat.get("id")
            if cid:
                targets.append(f"publication_target={cid}")
    return "&".join(targets)


_ANSWER_FIELDS = ("id", "nick", "status", "task_id", "answers", "duration", "revised_request_count")


def _build_apply_url(task_id: int, room_name: str, answer_id: Optional[int] = None) -> str:
    """Apply URL exactly like the official client.

    Draft tasks must include answer_id + answer_fields: short apply returns
    answer=None and the later PUT/POST then fails with 400. Fresh tasks use
    the short form without answer_id.
    """
    base = f"{IPTV_BASE_URL}/tms/task/{task_id}/apply"
    if answer_id:
        fields = "&".join(f"answer_fields={f}" for f in _ANSWER_FIELDS)
        return (
            f"{base}?preview_mode=false&answer_id={answer_id}"
            f"&{fields}&token_code=null&room_name={room_name}"
        )
    return f"{base}?preview_mode=false&token_code=null&room_name={room_name}"


def _decode_apply_response(resp) -> dict:
    """Apply returns JSON or base64-encoded JSON depending on the task."""
    try:
        return resp.json()
    except Exception:
        return json.loads(base64.b64decode(resp.text).decode("utf-8"))


async def _fetch_todo_page(client, headers: dict, targets_query: str, *, is_essay: bool,
                           expired_only: bool, filter_expired: bool,
                           answer_statuses: Optional[list] = None) -> list:
    essay_flag = "true" if is_essay else "false"
    url = (
        f"{IPTV_BASE_URL}/tms/task/todo?expired_only={'true' if expired_only else 'false'}"
        f"&limit=100&offset=0&filter_expired={'true' if filter_expired else 'false'}"
        f"&is_exam=false&with_answer=true&is_essay={essay_flag}"
        f"&{targets_query}&with_apply_moment=true"
    )
    if answer_statuses:
        url += "".join(f"&answer_statuses={s}" for s in answer_statuses)
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        logger.warning(f"[tasks] todo fetch failed HTTP {resp.status_code}: {url[:120]}")
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _merge_todo_pages(pages: list[list]) -> list:
    """Dedupe by task id, preferring entries that carry answer info (draft)."""
    merged: dict[int, dict] = {}
    for page in pages:
        for t in page or []:
            tid = t.get("id")
            if tid is None:
                continue
            prev = merged.get(tid)
            if prev is None:
                merged[tid] = t
                continue
            # Prefer draft/answer-carrying entry over bare pending (answer_status None).
            prev_score = 1 if prev.get("answer_status") == "draft" or prev.get("answer_id") else 0
            cur_score = 1 if t.get("answer_status") == "draft" or t.get("answer_id") else 0
            if cur_score > prev_score:
                merged[tid] = t
    return list(merged.values())


def _task_cache_key(user: dict, task_id: int) -> tuple[str, int]:
    return (user.get("username", ""), int(task_id))


def _snapshot_user_credentials(user: dict) -> dict:
    """Copy the fields background workers need.

    Workers must keep acting as the user who created the job even if that
    browser session logs out and someone else logs in afterwards, so they
    get a frozen copy instead of a live reference to the session dict.
    """
    return {
        "session_id": user.get("session_id"),
        "name": user.get("name"),
        "username": user.get("username"),
        "ra": user.get("ra"),
        "digito": user.get("digito"),
        "uf": user.get("uf"),
        "password": user.get("password"),
        "token_sed": user.get("token_sed"),
        "leiasp_jwt": user.get("leiasp_jwt"),
        "auth_token": user.get("auth_token"),
        "nick": user.get("nick"),
        "room_name": user.get("room_name", ""),
        "task_rooms": dict(user.get("task_rooms", {})),
        "apply_times": dict(user.get("apply_times", {})),
        "active_answer_ids": dict(user.get("active_answer_ids", {})),
    }


def _require_job_owner(job: Optional[dict], user: dict, label: str = "Job") -> dict:
    if not job:
        raise HTTPException(status_code=404, detail=f"{label} não encontrado")
    if job.get("username") != user.get("username"):
        raise HTTPException(status_code=403, detail="Não autorizado")
    return job

class LoginRequest(BaseModel):
    ra: str
    digito: str
    uf: str
    password: str

class SubmitRequest(BaseModel):
    answers: dict
    answer_id: Optional[int] = None
    duration: Optional[float] = None
    min_time: Optional[int] = None
    max_time: Optional[int] = None

class TaskBatchSolveRequest(BaseModel):
    task_ids: list[int]
    min_time: Optional[float] = None
    max_time: Optional[float] = None

class MatificCompleteRequest(BaseModel):
    episode: dict
    target_accuracy: Optional[str] = "realistic"
    timings: Optional[dict] = None

class MatificBatchRequest(BaseModel):
    episodes: list[dict]
    min_time_per_task: Optional[float] = 1.0
    max_time_per_task: Optional[float] = 3.0
    min_wait_between: Optional[float] = 0.2
    max_wait_between: Optional[float] = 1.0
    target_accuracy: Optional[str] = "realistic"

class MatificPurchaseRequest(BaseModel):
    campaign_id: str
    item_id: str
    cost: int

class MatificEquipRequest(BaseModel):
    campaign_id: str
    part_name: str
    item_id: str

class MatificRepairRequest(BaseModel):
    campaign_id: str

class MatificSetStatsRequest(BaseModel):
    campaign_id: str
    coins: Optional[int] = None
    xp: Optional[int] = None
    rank: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def index_page():
    tpl_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.isfile(tpl_path):
        with open(tpl_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SalaDoPassado Backend Online</h1>"

@app.post("/api/login")
async def login(req: LoginRequest, response: Response):
    ra_clean = req.ra.strip().lstrip("0") or req.ra.strip()
    digito_clean = req.digito.strip()
    uf_clean = req.uf.strip().upper()
    constructed_user = f"{ra_clean}{digito_clean}{uf_clean}"

    fp = _get_browser_context()
    headers_login = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9",
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": SED_SUBSCRIPTION_KEY,
        "X-Product-Name": "SalaDoFuturo",
        "Origin": "https://saladofuturo.educacao.sp.gov.br",
        "Referer": "https://saladofuturo.educacao.sp.gov.br/",
        "User-Agent": fp["user-agent"],
        "sec-ch-ua": fp["sec-ch-ua"],
        "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
        "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
    }

    async with HttpCloakClient(timeout=20.0) as client:
        resp_sed = await client.post(
            SED_LOGIN_URL,
            json={"user": constructed_user, "senha": req.password},
            headers=headers_login
        )
        if resp_sed.status_code != 200:
            raise HTTPException(status_code=400, detail="Credenciais inválidas ou erro no servidor da SED.")

        data_sed = resp_sed.json()
        token_sed = data_sed.get("token")
        if not token_sed:
            raise HTTPException(status_code=400, detail="Usuário ou senha incorretos.")

        user_info = data_sed.get("DadosUsuario", {})
        if isinstance(user_info, str):
            try:
                parsed_user = json.loads(user_info)
                user_info = parsed_user if isinstance(parsed_user, dict) else {}
            except (ValueError, TypeError):
                user_info = {}
        if not isinstance(user_info, dict):
            user_info = {}
        student_name = user_info.get("NAME") or user_info.get("Nome") or "Estudante"
        def _student_id_from_login(payload):
            """Obtém somente identificadores que o login ROTULA como código de aluno.

            Alguns provedores serializam DadosUsuario como JSON dentro de uma string.
            Um CD_USUARIO não é convertido em AlunoId por suposição.
            """
            student_fields = {"codigoaluno", "alunoid", "idaluno", "cdaluno", "codaluno"}
            def walk(obj, depth=0):
                if depth > 7:
                    return None
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        normalized = "".join(ch for ch in str(key).casefold() if ch.isalnum())
                        if normalized in student_fields and isinstance(value, (str, int)):
                            candidate = str(value).strip()
                            if candidate.isascii() and candidate.isdecimal():
                                return candidate
                    for value in obj.values():
                        found = walk(value, depth + 1)
                        if found:
                            return found
                elif isinstance(obj, (list, tuple)):
                    for value in obj[:50]:
                        found = walk(value, depth + 1)
                        if found:
                            return found
                elif isinstance(obj, str):
                    raw = obj.strip()
                    if len(raw) <= 50000 and raw.startswith(("{", "[")):
                        try:
                            return walk(json.loads(raw), depth + 1)
                        except (ValueError, TypeError):
                            pass
                return None
            return walk(payload)

        def _safe_field_names(obj):
            """Apenas nomes de campos (sem valores/segredos) para diagnóstico."""
            if not isinstance(obj, dict):
                return []
            # Evita indicar até mesmo nomes de campos que possam conter segredos.
            blocked = ("token", "senha", "password", "secret", "authorization", "cookie", "cpf", "ra")
            return sorted(str(k)[:70] for k in obj if not any(t in str(k).casefold() for t in blocked))[:70]

        student_id = _student_id_from_login(data_sed)
        def _authenticated_user_code(jwt):
            try:
                section = jwt.split(".")[1]
                content = base64.urlsafe_b64decode(section + "=" * (-len(section) % 4))
                claims = json.loads(content)
                code = claims.get("CD_USUARIO")
                return str(code) if str(code).isascii() and str(code).isdecimal() else None
            except (ValueError, TypeError, IndexError, UnicodeDecodeError):
                return None
        user_code = _authenticated_user_code(token_sed)
        # Os nomes dos campos ajudam no suporte, sem persistir o payload/token.
        login_field_names = _safe_field_names(data_sed)
        login_user_field_names = _safe_field_names(user_info)
        login_claim_field_names = []
        if not student_id:
            try:
                section = token_sed.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(section + "=" * (-len(section) % 4)))
                # Only explicit student-labelled claims count, never CD_USUARIO.
                student_id = _student_id_from_login(claims)
                login_claim_field_names = _safe_field_names(claims)
            except (ValueError, TypeError, IndexError, UnicodeDecodeError):
                pass

        headers_valida = dict(headers_login)
        headers_valida["Authorization"] = f"Bearer {token_sed}"
        await client.post(SED_VALIDA_URL, headers=headers_valida)

        leiasp_jwt = None
        try:
            url_leiasp = "https://sedintegracoes.educacao.sp.gov.br/saladofuturobffapi/integracoes/Token?plataforma=LeiaSP%2B"
            resp_tok = await client.get(url_leiasp, headers=headers_valida)
            if resp_tok.status_code == 200:
                leiasp_jwt = resp_tok.json().get("data")
        except Exception as e:
            logger.warning(f"[Login] Falha ao pré-obter LeiaSP JWT: {e}")

        headers_iptv = {
            "Content-Type": "application/json",
            "Accept-Language": "pt-BR,pt;q=0.9",
            "Origin": "https://saladofuturo.educacao.sp.gov.br",
            "Referer": "https://saladofuturo.educacao.sp.gov.br/",
            "User-Agent": fp["user-agent"],
            "sec-ch-ua": fp["sec-ch-ua"],
            "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
            "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
            "x-api-platform": "webclient",
            "x-api-realm": "edusp"
        }
        resp_iptv = await client.post(IPTV_TOKEN_URL, json={"token": token_sed}, headers=headers_iptv)
        if resp_iptv.status_code != 200:
            raise HTTPException(status_code=502, detail="Erro ao sincronizar token com a plataforma IPTV.")

        iptv_data = resp_iptv.json()
        # Alguns logins retornam o AlunoId apenas na resposta autenticada da plataforma.
        # Nunca confundir CD_USUARIO com AlunoId.
        if not student_id:
            student_id = _student_id_from_login(iptv_data)
        # Nomes escolares só podem vir de campos explicitamente rotulados na
        # autenticação (nunca de IDs, nomes de usuários ou dados de exemplo).
        def _school_label(payload, exact_fields):
            if not isinstance(payload, dict):
                return None
            wanted = {"".join(c for c in k.casefold() if c.isalnum()) for k in exact_fields}
            def visit(node, depth=0):
                if depth > 4:
                    return None
                if isinstance(node, dict):
                    for k, v in node.items():
                        key = "".join(c for c in str(k).casefold() if c.isalnum())
                        if key in wanted and isinstance(v, str):
                            label = v.strip()
                            if 3 <= len(label) <= 120 and not label.isdecimal():
                                return label
                    for v in node.values():
                        if isinstance(v, (dict, list)):
                            found = visit(v, depth + 1)
                            if found:
                                return found
                elif isinstance(node, list):
                    for v in node[:12]:
                        found = visit(v, depth + 1)
                        if found:
                            return found
                return None
            return visit(payload)
        school_name = (_school_label(data_sed, ("nomeEscola", "nmEscola", "outNomeEscola", "nomeUnidadeEscolar"))
                       or _school_label(iptv_data, ("nomeEscola", "nmEscola", "outNomeEscola", "nomeUnidadeEscolar")))
        class_name = (_school_label(data_sed, ("nomeTurma", "nmTurma", "outNomeTurma", "descricaoTurma"))
                      or _school_label(iptv_data, ("nomeTurma", "nmTurma", "outNomeTurma", "descricaoTurma")))
        auth_token = iptv_data.get("auth_token")
        nick = iptv_data.get("nick")

        if not auth_token or not nick:
            raise HTTPException(status_code=502, detail="Não foi possível obter os tokens da plataforma.")

        session_id = uuid.uuid4().hex
        user_session = {
            "session_id": session_id,
            "created_at": time.time(),
            "name": student_name,
            "student_id": student_id,
            "school_name": school_name,
            "school_class_name": class_name,
            "login_field_names": login_field_names,
            "login_user_field_names": login_user_field_names,
            "login_claim_field_names": login_claim_field_names,
            "sed_user_code": user_code,
            "username": constructed_user,
            "ra": ra_clean,
            "digito": digito_clean,
            "uf": uf_clean,
            "password": req.password,
            "token_sed": token_sed,
            "leiasp_jwt": leiasp_jwt,
            "auth_token": auth_token,
            "nick": nick,
            "room_name": "",
            "task_rooms": {},
            "apply_times": {},
            "active_answer_ids": {},
        }
        _sessions[session_id] = user_session
        now = time.time()
        for sid, sess in list(_sessions.items()):
            if now - sess.get("created_at", now) > _SESSION_TTL:
                _sessions.pop(sid, None)

        asyncio.create_task(_prewarm_matific_session(_snapshot_user_credentials(user_session)))

        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=86400
        )

        return {
            "success": True,
            "token": session_id,
            "user": {
                "name": student_name,
                "ra": ra_clean,
                "digito": digito_clean,
                "uf": uf_clean,
                "nick": nick
            }
        }

# Perfil escolar exclusivamente da sessão autenticada.
# Não incluir RA, escola ou turma fornecidos em imagens como dados fixos.
@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "name": user["name"],
        "ra": user["ra"],
        "digito": user["digito"],
        "uf": user["uf"],
        "nick": user["nick"],
        "student_code_available": bool(user.get("student_id")),
        "user_code_available": bool(user.get("sed_user_code")),
        "school": user.get("school_name"),
        "class_name": user.get("school_class_name"),
        "school_data_source": ("api" if user.get("school_name") or user.get("school_class_name") else "indisponivel"),
        "school_series": user.get("school_series"),
        "school_year": user.get("school_year"),
        "class_code_available": bool(user.get("school_class_code")),
        "school_class_code": user.get("school_class_code"),
        "school_code": user.get("school_code")
    }

@app.post("/api/logout")
async def logout(response: Response, user: dict = Depends(get_current_user)):
    sess = _sessions.pop(user.get("session_id"), None)
    username = (sess or user).get("username")
    if username:
        _matific_session_cache.pop(username, None)
        for key in [k for k in _task_cache if k[0] == username]:
            _task_cache.pop(key, None)
    response.delete_cookie("session_id")
    return {"success": True}


# Contagem do painel: consulta separada da listagem completa de tarefas.
# Nunca apresenta quantidade estimada como pendencias reais.
def _extract_official_pending_count(payload):
    if isinstance(payload, bool):
        return None
    if isinstance(payload, int) and payload >= 0:
        return payload
    if not isinstance(payload, dict):
        return None
    # Aceitar somente campos de contagem explicitos, nao tamanho de listas.
    for field in ("count", "total", "total_count", "totalCount", "pending_count", "pendingCount"):
        value = payload.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for field in ("data", "result"):
        nested = payload.get(field)
        if isinstance(nested, dict):
            found = _extract_official_pending_count(nested)
            if found is not None:
                return found
    return None

# A lista de tarefas e redacoes ja faz as consultas que o painel original utiliza.
# Fallback para o caso de o endpoint extra de contagem nao estar disponivel.
def _hyper_pending_item(item):
    if not isinstance(item, dict):
        return False
    status = str(item.get("answer_status") or "").lower()
    if status in ("finished", "submitted", "completed", "done", "expired"):
        return False
    if item.get("task_expired") is True or item.get("expired") is True:
        return False
    from datetime import datetime, timezone
    for field in ("expired_at", "expires_at", "expiration_date", "deadline"):
        value = item.get(field)
        if not value:
            continue
        try:
            date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            if date < datetime.now(timezone.utc):
                return False
        except (ValueError, TypeError, OverflowError):
            # Um prazo desconhecido nao prova que a tarefa expirou.
            pass
    return True

@app.get("/api/tasks/pending-summary")
async def get_hyper_pending_summary(user: dict = Depends(get_current_user)):
    """Usa a listagem existente de atividades para recuperar o contador do painel.

    Nao cria numero fixo, nao inclui expiradas ou concluidas explicitamente,
    e retorna erro se a listagem escolar nao puder ser consultada.
    """
    data = await list_tasks(user=user)
    tasks = data.get("tasks", [])
    essays = data.get("essays", [])
    if not isinstance(tasks, list) or not isinstance(essays, list):
        raise HTTPException(status_code=502, detail="Formato de atividades inesperado")
    active_tasks = [x for x in tasks if _hyper_pending_item(x)]
    active_essays = [x for x in essays if _hyper_pending_item(x)]
    return {
        "available": True,
        "count": len(active_tasks) + len(active_essays),
        "tasks_count": len(active_tasks),
        "essays_count": len(active_essays),
        "source": "api/tasks",
        "note": "Atividades nao expiradas e nao concluidas identificadas na listagem",
    }

@app.get("/api/tasks/pending-count")
async def get_official_pending_count(user: dict = Depends(get_current_user)):
    """Fetch the school task count, rather than adding all tasks returned by /api/tasks.

    Missing or unexpected responses are marked unavailable, never replaced by guesses.
    """
    fp = _get_browser_context()
    headers = {
        "x-api-key": user["auth_token"],
        "x-api-platform": "webclient",
        "x-api-realm": "edusp",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": fp["user-agent"],
    }
    async with HttpCloakClient(timeout=20.0) as client:
        rooms_resp = await client.get(f"{IPTV_BASE_URL}/room/user?list_all=true&with_cards=true", headers=headers)
        if rooms_resp.status_code != 200:
            return {"available": False, "reason": "Nao foi possivel consultar as salas"}
        rooms = rooms_resp.json().get("rooms", [])
        targets = _tms_targets_query(rooms, user.get("nick", ""))
        if not targets:
            return {"available": False, "reason": "Nenhuma sala encontrada"}
        # O app oficial usa filtros separados para a contagem de atividades.
        # Nao incluir tarefas expiradas nem concluidas.
        url = (f"{IPTV_BASE_URL}/tms/task/todo/count?{targets}"
               "&filter_expired=true&with_answer=true"
               "&answer_statuses=draft&answer_statuses=pending")
        try:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                return {"available": False, "reason": "A API de contagem nao respondeu"}
            count = _extract_official_pending_count(response.json())
            if count is None:
                return {"available": False, "reason": "Formato de contagem nao confirmado"}
            return {"available": True, "count": count, "source": "tms/task/todo/count"}
        except Exception:
            return {"available": False, "reason": "Nao foi possivel interpretar a contagem"}

@app.get("/api/tasks")
async def list_tasks(user: dict = Depends(get_current_user)):
    fp = _get_browser_context()
    headers_tms = {
        "x-api-key": user["auth_token"],
        "x-api-platform": "webclient",
        "x-api-realm": "edusp",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9",
        "User-Agent": fp["user-agent"],
        "sec-ch-ua": fp["sec-ch-ua"],
        "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
        "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
    }

    async with HttpCloakClient(timeout=20.0) as client:
        url_rooms = f"{IPTV_BASE_URL}/room/user?list_all=true&with_cards=true"
        resp_rooms = await client.get(url_rooms, headers=headers_tms)
        if resp_rooms.status_code != 200:
            raise HTTPException(status_code=502, detail="Erro ao consultar salas do aluno.")

        rooms = resp_rooms.json().get("rooms", [])
        room_names = {r.get("name") for r in rooms if r.get("name")}
        if rooms:
            user["room_name"] = rooms[0].get("name", user.get("room_name", ""))

        targets_query = _tms_targets_query(rooms, user.get("nick", ""))

        # Official client flow (harfiles/tarefas_update.har):
        # - pending non-expired: no answer_statuses filter
        # - drafts non-expired: answer_statuses=draft
        # - expired (pending + drafts): expired_only=true + answer_statuses=draft
        tasks_pages = [
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=False,
                                   expired_only=False, filter_expired=True, answer_statuses=None),
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=False,
                                   expired_only=False, filter_expired=True, answer_statuses=["draft"]),
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=False,
                                   expired_only=True, filter_expired=False, answer_statuses=["draft"]),
        ]
        essays_pages = [
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=True,
                                   expired_only=False, filter_expired=True, answer_statuses=None),
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=True,
                                   expired_only=False, filter_expired=True, answer_statuses=["draft"]),
            await _fetch_todo_page(client, headers_tms, targets_query, is_essay=True,
                                   expired_only=True, filter_expired=False, answer_statuses=["draft"]),
        ]

        tasks_raw = _merge_todo_pages(tasks_pages)
        essays_raw = _merge_todo_pages(essays_pages)

        # Apply must use the room the task was published to; rooms[0] returns
        # 403 {"field":"task","cause":"denied"} for tasks from other rooms.
        task_rooms = user.setdefault("task_rooms", {})
        for t in tasks_raw + essays_raw:
            tid = t.get("id")
            troom = t.get("publication_target")
            if tid and troom in room_names:
                task_rooms[int(tid)] = troom

        tasks_pending = [t for t in tasks_raw if t.get("answer_status") not in ("finished", "submitted")]
        essays_pending = [t for t in essays_raw if t.get("answer_status") not in ("finished", "submitted")]

    return {
        "tasks": tasks_pending,
        "essays": essays_pending
    }

async def _discover_task_meta(client, headers_tms: dict, targets_query: str, task_id: int) -> tuple[Optional[str], Optional[int]]:
    """Find a task's publication room and draft answer_id by scanning todo pages.

    The same task can be listed as fresh/pending or as a draft, so both plain
    and draft-filtered pages are scanned (fresh + expired).
    """
    room: Optional[str] = None
    answer_id: Optional[int] = None

    async def scan(is_essay: bool, expired_only: bool, filter_expired: bool, statuses):
        nonlocal room, answer_id
        page = await _fetch_todo_page(
            client, headers_tms, targets_query, is_essay=is_essay,
            expired_only=expired_only, filter_expired=filter_expired,
            answer_statuses=statuses,
        )
        for t in page:
            if t.get("id") != task_id:
                continue
            if not room and t.get("publication_target"):
                room = t["publication_target"]
            if not answer_id and t.get("answer_id"):
                try:
                    answer_id = int(t["answer_id"])
                except Exception:
                    answer_id = t["answer_id"]

    for is_essay in (False, True):
        await scan(is_essay, False, True, None)
        await scan(is_essay, False, True, ["draft"])
        await scan(is_essay, True, False, ["draft"])
    return room, answer_id

@app.get("/api/task/{task_id}")
async def get_task_detail(task_id: int, answer_id: Optional[int] = None, user: dict = Depends(get_current_user)):
    async with _task_preparation_semaphore:
        return await _get_task_detail(task_id, answer_id=answer_id, user=user)


async def _get_task_detail(task_id: int, *, user: dict, answer_id: Optional[int] = None):
    """Open a task while the caller holds a preparation slot."""
    fp = _get_browser_context()
    headers_tms = {
        "x-api-key": user["auth_token"],
        "x-api-platform": "webclient",
        "x-api-realm": "edusp",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "pt-BR,pt;q=0.9",
        "User-Agent": fp["user-agent"],
        "sec-ch-ua": fp["sec-ch-ua"],
        "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
        "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
    }

    room_name = user.get("task_rooms", {}).get(task_id) or user.get("room_name", "")
    resolved_answer_id = answer_id or user.get("active_answer_ids", {}).get(task_id)

    async with HttpCloakClient(timeout=30.0) as client:
        url_apply = _build_apply_url(task_id, room_name, resolved_answer_id)
        resp = await tms_apply_with_captcha(client, url_apply, headers_tms, task_id)
        task_data = None
        if resp.status_code == 200:
            try:
                task_data = _decode_apply_response(resp)
            except Exception:
                task_data = None
        # 403 {"field":"task","cause":"denied"} means the task belongs to a room
        # other than the one used (the official client opens it from its room).
        room_denied = resp.status_code == 403 and "captcha" not in (resp.text or "").lower()
        needs_discovery = room_denied or (
            not resolved_answer_id
            and (resp.status_code == 400 or (task_data is not None and task_data.get("answer") is None))
        )
        if needs_discovery:
            # Short apply loses the draft (answer=None) and the later PUT/POST fails
            # with 400. Discover the task's room/answer_id via todo and retry.
            try:
                url_rooms = f"{IPTV_BASE_URL}/room/user?list_all=true&with_cards=true"
                resp_rooms = await client.get(url_rooms, headers=headers_tms)
                rooms = resp_rooms.json().get("rooms", []) if resp_rooms.status_code == 200 else []
                targets_query = _tms_targets_query(rooms, user.get("nick", ""))
                found_room, found_answer_id = await _discover_task_meta(client, headers_tms, targets_query, task_id)
                if found_room:
                    room_name = found_room
                    user.setdefault("task_rooms", {})[task_id] = found_room
                if found_answer_id and not resolved_answer_id:
                    resolved_answer_id = found_answer_id
                if found_room or found_answer_id:
                    url_apply = _build_apply_url(task_id, room_name, resolved_answer_id)
                    resp = await tms_apply_with_captcha(client, url_apply, headers_tms, task_id)
                    task_data = None
            except Exception as e:
                logger.warning(f"[task-detail] task meta discovery failed for {task_id}: {e}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Erro ao abrir tarefa (HTTP {resp.status_code})")

        if task_data is None:
            task_data = _decode_apply_response(resp)

    _task_cache[_task_cache_key(user, task_id)] = task_data
    user["apply_times"][task_id] = time.time()
    
    ans_obj = task_data.get("answer")
    if isinstance(ans_obj, dict) and ans_obj.get("id"):
        user["active_answer_ids"][task_id] = ans_obj["id"]

    return task_data

@app.post("/api/ai-fill/{task_id}")
async def ai_fill_task(task_id: int, user: dict = Depends(get_current_user)):
    async with _task_preparation_semaphore:
        task_data = _task_cache.get(_task_cache_key(user, task_id))
        if not task_data:
            task_data = await _get_task_detail(task_id, user=user)
        return await resolve_task_answers(task_data, task_id)

@app.post("/api/task/{task_id}/submit")
async def submit_task(task_id: int, req: SubmitRequest, user: dict = Depends(get_current_user)):
    task_data = _task_cache.get(_task_cache_key(user, task_id))
    if not task_data:
        task_data = await get_task_detail(task_id, user=user)

    is_essay = task_data.get("is_essay") or task_data.get("task_is_essay") or any(
        q.get("type") == "essay" for q in task_data.get("questions", [])
    )

    answers = req.answers
    for entry in answers.values():
        if isinstance(entry, dict) and entry.get("question_type") == "fill-letters":
            ans_v = entry.get("answer")
            if isinstance(ans_v, list):
                entry["answer"] = "".join(str(x) for x in ans_v)

    apply_time = user.get("apply_times", {}).get(task_id)
    now = time.time()
    if apply_time:
        duration = round(now - apply_time, 2)
    else:
        duration = req.duration or 120.0

    if req.min_time and req.max_time:
        duration = random.uniform(req.min_time * 60, req.max_time * 60)
    elif duration < 60.0:
        duration = 60.0 + random.uniform(5.0, 30.0)

    room_name = user.get("task_rooms", {}).get(task_id) or user.get("room_name", "")
    answer_id = req.answer_id or user.get("active_answer_ids", {}).get(task_id)

    submit_status = "draft" if is_essay else "submitted"
    submit_payload = {
        "status": submit_status,
        "answers": answers,
        "accessed_on": "room",
        "executed_on": room_name,
        "duration": duration
    }

    fp = _get_browser_context()
    traceparent, request_id = _generate_traceparent()
    headers_submit = {
        "accept": "application/json",
        "accept-language": "pt-BR,pt;q=0.9",
        "content-type": "application/json",
        "origin": "https://saladofuturo.educacao.sp.gov.br",
        "referer": "https://saladofuturo.educacao.sp.gov.br/",
        "request-id": request_id,
        "sec-ch-ua": fp["sec-ch-ua"],
        "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
        "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "traceparent": traceparent,
        "user-agent": fp["user-agent"],
        "x-api-key": user["auth_token"]
    }

    async with HttpCloakClient(timeout=30.0) as client:
        if is_essay:
            essay_title = ""
            essay_body = ""
            for entry in answers.values():
                if isinstance(entry, dict) and entry.get("question_type") == "essay":
                    ans = entry.get("answer", {})
                    if isinstance(ans, dict):
                        essay_title = ans.get("title", "")
                        essay_body = ans.get("body", "")
                        break
            try:
                check_payload = {"title": essay_title, "body": essay_body}
                await client.post(
                    f"{IPTV_BASE_URL}/tms/task/{task_id}/essay-check",
                    json=check_payload,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/json",
                        "origin": "https://saladofuturo.educacao.sp.gov.br",
                        "referer": "https://saladofuturo.educacao.sp.gov.br/",
                        "user-agent": fp["user-agent"],
                        "x-api-key": user["auth_token"]
                    }
                )
            except Exception as e:
                logger.warning(f"essay-check warning: {e}")

        # Refresh answer_id via apply using the HAR format (answer_id + answer_fields when known).
        # Short apply on a draft returns answer=None, then PUT/POST fails with 400,
        # so always include answer_id if we have it.
        url_apply = _build_apply_url(task_id, room_name, answer_id)
        headers_tms = {
            "x-api-key": user["auth_token"],
            "x-api-platform": "webclient",
            "x-api-realm": "edusp",
            "Accept": "application/json",
            "User-Agent": fp["user-agent"],
        }
        async with _task_preparation_semaphore:
            apply_resp = await tms_apply_with_captcha(client, url_apply, headers_tms, task_id)
        if apply_resp.status_code == 200:
            try:
                apply_data = _decode_apply_response(apply_resp)
                if isinstance(apply_data.get("answer"), dict) and apply_data["answer"].get("id"):
                    answer_id = apply_data["answer"]["id"]
            except Exception:
                pass

        if answer_id:
            url_sub = f"{IPTV_BASE_URL}/tms/task/{task_id}/answer/{answer_id}"
            resp_sub = await client.put(url_sub, json=submit_payload, headers=headers_submit)
        else:
            url_sub = f"{IPTV_BASE_URL}/tms/task/{task_id}/answer"
            resp_sub = await client.post(url_sub, json=submit_payload, headers=headers_submit)

        if resp_sub.status_code != 200:
            raise HTTPException(status_code=resp_sub.status_code, detail=f"Falha no envio: {resp_sub.text[:300]}")

        result_data = resp_sub.json() if resp_sub.text else {}
        user.get("apply_times", {}).pop(task_id, None)

        return {
            "success": True,
            "task_id": task_id,
            "answer_id": result_data.get("id", answer_id),
            "status": result_data.get("status", submit_status),
            "score": result_data.get("result_score"),
            "message": "Enviado com sucesso!" if not is_essay else "Redação salva com sucesso (Rascunho)!"
        }

async def _delayed_worker(job_id: str, task_id: int, req: SubmitRequest, user: dict, delay_seconds: float):
    _delayed_jobs[job_id]["status"] = "waiting"
    _delayed_jobs[job_id]["remaining_seconds"] = delay_seconds
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = max(0, delay_seconds - elapsed)
        _delayed_jobs[job_id]["remaining_seconds"] = round(remaining)
        if remaining <= 0:
            break
        await asyncio.sleep(min(1.0, remaining))

    _delayed_jobs[job_id]["status"] = "submitting"
    try:
        res = await submit_task(task_id, req, user)
        _delayed_jobs[job_id]["status"] = "finished"
        _delayed_jobs[job_id]["result"] = res
    except Exception as e:
        _delayed_jobs[job_id]["status"] = "error"
        _delayed_jobs[job_id]["error"] = str(e)

@app.post("/api/task/{task_id}/submit-delayed")
async def submit_delayed(task_id: int, req: SubmitRequest, user: dict = Depends(get_current_user)):
    job_id = uuid.uuid4().hex
    num_q = len(req.answers)
    
    if req.min_time and req.max_time:
        delay = random.uniform(req.min_time * 60, req.max_time * 60)
    else:
        delay = num_q * 90.0 + random.randint(10, 45)

    _delayed_jobs[job_id] = {
        "job_id": job_id,
        "username": user["username"],
        "task_id": task_id,
        "status": "queued",
        "remaining_seconds": round(delay),
        "created_at": time.time()
    }

    asyncio.create_task(_delayed_worker(job_id, task_id, req, _snapshot_user_credentials(user), delay))

    return {
        "success": True,
        "job_id": job_id,
        "remaining_seconds": round(delay),
        "message": f"Envio agendado para daqui a {round(delay / 60, 1)} minutos."
    }

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    job = _delayed_jobs.get(job_id)
    _require_job_owner(job, user)
    return job

async def _solve_single_task_worker(batch_id: str, task_id: int, user: dict, min_time: Optional[float], max_time: Optional[float]):
    batch = _task_batches.get(batch_id)
    if not batch or task_id not in batch["tasks"]:
        return

    task_info = batch["tasks"][task_id]
    task_info["status"] = "queued"
    task_info["message"] = "Na fila para abrir a tarefa e consultar a IA..."

    try:
        async with _task_preparation_semaphore:
            if batch.get("status") == "stopped":
                task_info["status"] = "stopped"
                task_info["message"] = "Execução cancelada pelo usuário."
                return
            task_info["status"] = "resolving_ai"
            task_info["message"] = "Abrindo tarefa e resolvendo questões com IA..."
            task_data = await _get_task_detail(task_id, user=user)
            task_info["title"] = task_data.get("title", task_info.get("title", f"Tarefa #{task_id}"))
            if batch.get("status") == "stopped":
                task_info["status"] = "stopped"
                task_info["message"] = "Execução cancelada pelo usuário."
                return
            ai_res = await resolve_task_answers(task_data, task_id)

        if not ai_res.get("success") or not ai_res.get("answers"):
            task_info["status"] = "failed"
            task_info["message"] = "IA não gerou respostas para a tarefa."
            return

        answers = ai_res["answers"]
        questions = [q for q in task_data.get("questions", []) if q.get("type") not in ("info", "section")]
        num_q = max(1, len(questions))

        if min_time and max_time:
            delay = random.uniform(min_time * 60, max_time * 60)
        else:
            delay = max(45.0, num_q * 90.0 + random.randint(10, 45))

        task_info["status"] = "waiting_delay"
        task_info["total_seconds"] = round(delay)
        task_info["remaining_seconds"] = round(delay)
        task_info["message"] = f"Aguardando tempo humanizado ({round(delay)}s)..."

        start_time = time.time()
        while True:
            if batch.get("status") == "stopped":
                task_info["status"] = "stopped"
                task_info["message"] = "Execução cancelada pelo usuário."
                return
            elapsed = time.time() - start_time
            rem = max(0, delay - elapsed)
            task_info["remaining_seconds"] = round(rem)
            if rem <= 0:
                break
            await asyncio.sleep(min(1.0, rem))

        if batch.get("status") == "stopped":
            task_info["status"] = "stopped"
            task_info["message"] = "Execução cancelada pelo usuário."
            return

        task_info["status"] = "submitting"
        task_info["message"] = "Enviando respostas..."

        submit_req = SubmitRequest(answers=answers, duration=delay)
        submit_res = await submit_task(task_id, submit_req, user)

        task_info["status"] = "completed"
        task_info["score"] = submit_res.get("score")
        task_info["message"] = submit_res.get("message", "Concluído com sucesso!")

    except asyncio.CancelledError:
        task_info["status"] = "stopped"
        task_info["message"] = "Execução cancelada."
        raise
    except Exception as e:
        task_info["status"] = "failed"
        task_info["message"] = f"Erro: {str(e)[:150]}"
    finally:
        completed = sum(1 for t in batch["tasks"].values() if t["status"] in ("completed", "failed", "stopped"))
        batch["completed_count"] = completed
        if completed >= batch["total"] and batch["status"] != "stopped":
            batch["status"] = "completed"

@app.post("/api/tasks/batch-solve")
async def start_tasks_batch_solve(req: TaskBatchSolveRequest, user: dict = Depends(get_current_user)):
    if not req.task_ids:
        raise HTTPException(status_code=400, detail="Nenhuma tarefa selecionada.")

    seen_ids: set[int] = set()
    task_ids: list[int] = []
    for tid in req.task_ids:
        try:
            tid_int = int(tid)
        except Exception:
            continue
        if tid_int not in seen_ids:
            seen_ids.add(tid_int)
            task_ids.append(tid_int)
    if not task_ids:
        raise HTTPException(status_code=400, detail="Nenhuma tarefa selecionada.")

    # A task already in flight in another active batch cannot join a new one.
    busy: dict[int, str] = {}
    for b in _task_batches.values():
        if b.get("username") != user["username"]:
            continue
        if b.get("status") not in ("running", "queued"):
            continue
        for tid, tinfo in (b.get("tasks") or {}).items():
            if (tinfo or {}).get("status") in ("queued", "resolving_ai", "waiting_delay", "submitting"):
                try:
                    busy[int(tid)] = b.get("id")
                except Exception:
                    pass
    conflicts = [tid for tid in task_ids if tid in busy]
    if conflicts:
        raise HTTPException(
            status_code=400,
            detail=(
                "Tarefa(s) já em execução em outro lote: "
                + ", ".join(str(t) for t in conflicts)
                + ". Aguarde concluir ou pare o lote."
            ),
        )

    batch_id = uuid.uuid4().hex
    tasks_map = {
        tid: {
            "id": tid,
            "title": f"Tarefa #{tid}",
            "status": "queued",
            "remaining_seconds": 0,
            "total_seconds": 0,
            "score": None,
            "message": "Na fila para abrir a tarefa e consultar a IA..."
        }
        for tid in task_ids
    }

    batch = {
        "id": batch_id,
        "username": user["username"],
        "total": len(task_ids),
        "completed_count": 0,
        "status": "running",
        "created_at": time.time(),
        "tasks": tasks_map
    }
    _task_batches[batch_id] = batch

    user_snapshot = _snapshot_user_credentials(user)
    for tid in task_ids:
        asyncio.create_task(_solve_single_task_worker(batch_id, tid, user_snapshot, req.min_time, req.max_time))

    return {
        "success": True,
        "batch_id": batch_id,
        "total": len(task_ids),
        "message": f"{len(task_ids)} tarefas na fila; até 5 em preparação simultânea."
    }

@app.get("/api/tasks/batch/{batch_id}")
async def get_tasks_batch_status(batch_id: str, user: dict = Depends(get_current_user)):
    batch = _task_batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Lote de tarefas não encontrado.")
    if batch.get("username") != user["username"]:
        raise HTTPException(status_code=403, detail="Não autorizado")
    return {"success": True, "batch": batch}

@app.get("/api/tasks/batches")
async def list_tasks_batches(user: dict = Depends(get_current_user)):
    """All batches of the current user (newest first) so the UI can track several in parallel."""
    username = user["username"]
    user_batches = [b for b in _task_batches.values() if b.get("username") == username]
    user_batches.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"success": True, "batches": user_batches}

@app.get("/api/tasks/active-batch")
async def get_active_tasks_batch(user: dict = Depends(get_current_user)):
    username = user["username"]
    user_batches = [b for b in _task_batches.values() if b.get("username") == username]
    if not user_batches:
        return {"active": False, "batch": None}
    user_batches.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    active = [b for b in user_batches if b.get("status") in ("running", "queued")]
    if active:
        return {"active": True, "batch": active[0]}
    recent = user_batches[0]
    if time.time() - recent.get("created_at", 0) < 30:
        return {"active": True, "batch": recent}
    return {"active": False, "batch": None}

@app.post("/api/tasks/batch/{batch_id}/stop")
async def stop_tasks_batch(batch_id: str, user: dict = Depends(get_current_user)):
    batch = _task_batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Lote de tarefas não encontrado.")
    if batch.get("username") != user["username"]:
        raise HTTPException(status_code=403, detail="Não autorizado")
    batch["status"] = "stopped"
    return {"success": True, "message": "Lote interrompido com sucesso."}

# ---------------------------------------------------------------------------
# Matific Helpers & Workers
# ---------------------------------------------------------------------------
def get_user_matific_batch_state(username: str):
    if username not in _matific_batches:
        _matific_batches[username] = {
            "status": "idle",
            "current_episode_slug": None,
            "current_episode_title": None,
            "current_progress_percent": 0.0,
            "completed_count": 0,
            "total_count": 0,
            "pending": [],
            "logs": [],
            "current_job_id": None
        }
    return _matific_batches[username]

def generate_matific_timings(min_task_minutes: float, max_task_minutes: float, problem_count: int) -> dict:
    target_duration = random.uniform(min_task_minutes * 60, max_task_minutes * 60)
    fixed_delays = 10.0
    overhead = (problem_count * 1.5) + (max(0, problem_count - 1) * 2.0)
    available_sec = max(20.0, target_duration - fixed_delays - overhead)
    sec_per_question = available_sec / max(1, problem_count)
    standard_avg = 30.0
    scale = max(0.1, min(5.0, sec_per_question / standard_avg))
    return {
        "loading_multiplier": 1.0,
        "solving_min": round(10.0 * scale, 1),
        "solving_max": round(25.0 * scale, 1),
        "struggle_min": round(5.0 * scale, 1),
        "struggle_max": round(15.0 * scale, 1),
        "reading_min": round(2.0 * scale, 1),
        "reading_max": round(6.0 * scale, 1),
        "inter_question_min": round(1.0 * scale, 1),
        "inter_question_max": round(3.0 * scale, 1)
    }

def _invalidate_matific_episodes_cache(username: str):
    if username in _matific_session_cache:
        _matific_session_cache[username]["episodes_assigned"] = None
        _matific_session_cache[username]["episodes_adventure"] = None
        _matific_session_cache[username]["state"] = None

async def _prewarm_matific_session(user: dict):
    try:
        client = await _get_matific_client(user)
        if client:
            try:
                await asyncio.gather(
                    get_matific_episodes("trabalho_atribuido", user),
                    get_matific_episodes("ilha_aventura", user),
                    get_matific_state(user),
                    return_exceptions=True
                )
            finally:
                await client.__aexit__(None, None, None)
    except Exception as e:
        logger.info(f"[MATIFIC-PREWARM] Background note: {e}")

async def _get_matific_client(user: dict) -> Optional[MatificClient]:
    username = user["username"]
    if username not in _matific_locks:
        _matific_locks[username] = asyncio.Lock()

    async with _matific_locks[username]:
        cached = _matific_session_cache.get(username)
        client_obj = MatificClient(
            ra=user["ra"],
            digito=user["digito"],
            uf=user["uf"],
            password=user["password"]
        )
        await client_obj.__aenter__()

        if cached and time.time() < cached["expires_at"]:
            for d in ("www.matific.com", "matific.com", ""):
                client_obj.client.set_cookie("sessionid", cached["sessionid"], domain=d)
                client_obj.client.set_cookie("csrftoken", cached["csrftoken"], domain=d)
                client_obj.client.set_cookie("user_data_token", cached["user_data_token"], domain=d)
                if cached.get("slatemath_user_id"):
                    client_obj.client.set_cookie("slatemath_user_id", cached["slatemath_user_id"], domain=d)
            client_obj.sessionid = cached["sessionid"]
            client_obj.csrftoken = cached["csrftoken"]
            client_obj.user_data_token = cached["user_data_token"]
            client_obj.slatemath_user_id = cached.get("slatemath_user_id")
            client_obj.init_data = cached.get("init_data")
            return client_obj

        auth_ok = await client_obj.authenticate()
        if not auth_ok:
            await client_obj.__aexit__(None, None, None)
            return None

        _matific_session_cache[username] = {
            "sessionid": client_obj.sessionid,
            "csrftoken": client_obj.csrftoken,
            "user_data_token": client_obj.user_data_token,
            "slatemath_user_id": client_obj.slatemath_user_id,
            "expires_at": time.time() + _MATIFIC_SESSION_TTL,
            "init_data": None,
            "episodes_assigned": None,
            "episodes_adventure": None,
            "state": None,
        }
        return client_obj

async def _matific_worker_task(username: str, user: dict, episode: dict, target_accuracy: str, job_id: str, timings: dict = None):
    job = _matific_jobs.get(job_id)
    if not job:
        return
    job["status"] = "running"
    job["logs"].append("Iniciando automação do Matific...")
    client = await _get_matific_client(user)
    if not client:
        job["status"] = "failed"
        job["logs"].append("Erro ao autenticar no Matific. Verifique as credenciais.")
        return

    try:
        job["logs"].append("Autenticação com sucesso! Iniciando simulação do episódio...")
        def on_progress(message, percent):
            if job["status"] == "stopped":
                raise asyncio.CancelledError("Stopped by user")
            job["logs"].append(f"[{percent}%] {message}")
            job["progress_percent"] = percent

        success = await client.complete_episode(episode, target_accuracy=target_accuracy, on_progress=on_progress, timings=timings)
        if success:
            job["status"] = "completed"
            job["logs"].append("Episódio concluído e enviado com sucesso!")
            job["progress_percent"] = 100
            _invalidate_matific_episodes_cache(username)
        else:
            job["status"] = "failed"
            job["logs"].append("O cliente não conseguiu concluir o episódio.")
    except asyncio.CancelledError:
        job["status"] = "stopped"
        job["logs"].append("Simulação interrompida pelo usuário.")
    except Exception as e:
        logger.error(f"[MATIFIC-WORKER] Error: {e}", exc_info=True)
        job["status"] = "failed"
        job["logs"].append(f"Erro na execução: {str(e)}")
    finally:
        await client.__aexit__(None, None, None)

async def _matific_batch_worker(username: str, user: dict, episodes: list, min_time_per_task: float, max_time_per_task: float, min_wait_between: float, max_wait_between: float, target_accuracy: str):
    state = get_user_matific_batch_state(username)
    state["status"] = "running"
    state["total_count"] = len(episodes)
    state["completed_count"] = 0
    state["pending"] = episodes.copy()
    state["logs"] = ["Iniciando lote de tarefas Matific..."]
    state["current_episode_slug"] = None
    state["current_episode_title"] = None
    state["current_progress_percent"] = 0.0
    state["current_job_id"] = None

    try:
        for idx, ep in enumerate(episodes):
            if state["status"] != "running":
                break
            state["current_episode_slug"] = ep.get("slug")
            state["current_episode_title"] = ep.get("title", ep.get("slug"))
            state["current_progress_percent"] = 0.0
            job_id = uuid.uuid4().hex
            state["current_job_id"] = job_id
            _matific_jobs[job_id] = {
                "status": "pending",
                "episode_slug": ep.get("slug"),
                "episode_title": ep.get("title", ep.get("slug")),
                "logs": ["Simulação individual do lote agendada."],
                "progress_percent": 0.0,
                "username": username
            }
            state["logs"].append(f"[{idx+1}/{len(episodes)}] Iniciando: {ep.get('title')}")
            problem_count = ep.get("problem_count", 5) or 5
            timings = generate_matific_timings(min_time_per_task, max_time_per_task, problem_count)
            await _matific_worker_task(username, user, ep, target_accuracy, job_id, timings=timings)

            job_res = _matific_jobs.get(job_id)
            if job_res and job_res["status"] == "completed":
                state["completed_count"] += 1
                state["logs"].append(f"[{idx+1}/{len(episodes)}] Concluído com sucesso: {ep.get('title')}")
            else:
                reason = job_res["logs"][-1] if job_res and job_res["logs"] else "Erro desconhecido"
                state["logs"].append(f"[{idx+1}/{len(episodes)}] Falhou: {ep.get('title')} ({reason})")

            if state["pending"]:
                state["pending"].pop(0)

            if idx < len(episodes) - 1 and state["status"] == "running":
                wait_sec = random.uniform(min_wait_between * 60, max_wait_between * 60)
                state["logs"].append(f"Aguardando {int(wait_sec)}s antes de iniciar a próxima tarefa...")
                sleep_start = time.time()
                while time.time() - sleep_start < wait_sec:
                    if state["status"] != "running":
                        break
                    await asyncio.sleep(1.0)
    except Exception as e:
        logger.error(f"[MATIFIC-BATCH] Error in batch worker: {e}", exc_info=True)
        state["logs"].append(f"Erro no executor do lote: {str(e)}")
        state["status"] = "failed"
    finally:
        _invalidate_matific_episodes_cache(username)
        # Keep the terminal status (completed/stopped/failed) visible so the
        # frontend polling /api/matific/batch/status can finish the progress
        # UI; only clear the live pointers.
        if state["status"] == "running":
            state["status"] = "completed"
        state["current_episode_slug"] = None
        state["current_episode_title"] = None
        state["current_progress_percent"] = 0.0
        state["current_job_id"] = None

# ---------------------------------------------------------------------------
# Matific Routes
# ---------------------------------------------------------------------------
@app.get("/api/matific/episodes")
async def get_matific_episodes(source: str = "trabalho_atribuido", user: dict = Depends(get_current_user)):
    username = user["username"]
    cached_session = _matific_session_cache.get(username)
    cache_key = "episodes_assigned" if source == "trabalho_atribuido" else "episodes_adventure"
    if cached_session and cached_session.get(cache_key):
        ts, cached_list = cached_session[cache_key]
        if time.time() - ts < _MATIFIC_EPISODES_TTL and cached_list:
            return {"success": True, "episodes": cached_list, "source": source, "cached": True}

    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")

    try:
        init_data = getattr(client, "init_data", None)
        if not init_data:
            init_data = await client.get_init_data()
            if init_data and username in _matific_session_cache:
                _matific_session_cache[username]["init_data"] = init_data

        if not init_data:
            raise HTTPException(status_code=502, detail="Falha ao obter dados de inicialização do Matific")

        episodes_list = []
        campaigns = init_data.get("Campaigns", [])
        campaign_ids = [c.get("Id") for c in campaigns if c.get("Id")]
        default_campaign_id = campaign_ids[0] if campaign_ids else None

        if source == "ilha_aventura":
            enrichment = await client.get_episode_enrichment()
            game_state = await client.fetch_game_state(campaign_ids=campaign_ids or [])
            played_episodes = _aggregate_played_episodes(game_state)

            road_seen = set()
            for ent in game_state.get("game_entity", []):
                if ent.get("object_type") != "Matific.Mad.EpisodeStorableData":
                    continue
                ent_id = ent.get("entity_id")
                if ent_id in road_seen:
                    continue
                road_seen.add(ent_id)
                agg = played_episodes.get(ent_id)
                if agg and agg.get("was_passed"):
                    continue

                assigned_ids = set()
                for c in campaigns:
                    for ep in c.get("Episodes", []):
                        assigned_ids.add(ep.get("EpisodeId"))
                for key in ["School", "Home", "Parent"]:
                    for ass in init_data.get("Assignments", {}).get(key, []):
                        assigned_ids.add(ass.get("EpisodeId"))
                if ent_id in assigned_ids:
                    continue

                meta = enrichment.get(ent_id, {})
                slug = meta.get("Slug")
                if not slug:
                    continue

                title = meta.get("Title") or translate_matific_slug(slug)
                subtitle = meta.get("Subtitle", "")

                ep_data = {
                    "slug": slug,
                    "assignment_id": ent.get("instance_id") or ent.get("row_id"),
                    "campaign_id": default_campaign_id,
                    "context_id": 13,
                    "activity_context": "13",
                    "assignment_type": 4,
                    "due_date": None,
                    "source": "Ilha da Aventura",
                    "episode_id": ent_id,
                    "title": title,
                    "subtitle": subtitle,
                    "problem_count": 5,
                    "completed": False,
                    "highest_score": ent.get("highest_score"),
                    "was_passed": False,
                    "zone": ent.get("zone"),
                    "order": ent.get("order")
                }
                episodes_list.append(ep_data)

            episodes_list.sort(key=lambda x: (x.get("zone") or 0, x.get("order") or 0))
            if username in _matific_session_cache:
                _matific_session_cache[username]["episodes_adventure"] = (time.time(), episodes_list)
            return {"success": True, "episodes": episodes_list, "source": source}

        played_episodes = {}
        if campaign_ids:
            try:
                game_state = await client.fetch_game_state(campaign_ids=campaign_ids)
                played_episodes = _aggregate_played_episodes(game_state)
            except Exception as ex:
                logger.error(f"[MATIFIC-EPISODES] Game state error: {ex}")

        for campaign in campaigns:
            campaign_id = campaign.get("Id")
            campaign_name = campaign.get("TranslatedName") or campaign.get("Name") or "Campanha Ativa"
            context_id = campaign.get("NewContext", 13)
            for ep in campaign.get("Episodes", []):
                episode_id = ep.get("EpisodeId")
                pe = played_episodes.get(episode_id)
                slug = ep.get("Slug")
                title = translate_matific_slug(slug)
                ep_data = {
                    "slug": slug,
                    "assignment_id": ep.get("AssignmentId"),
                    "campaign_id": campaign_id,
                    "context_id": context_id,
                    "activity_context": str(context_id),
                    "assignment_type": 4,
                    "due_date": ep.get("DueDate"),
                    "source": f"Campanha: {campaign_name}",
                    "episode_id": episode_id,
                    "title": title,
                    "subtitle": "",
                    "problem_count": 5,
                    "order": ep.get("Order") or 0,
                    "completed": pe is not None and (pe.get("was_passed") or pe.get("highest_score") is not None),
                    "highest_score": pe["highest_score"] if pe else None,
                    "was_passed": pe["was_passed"] if pe else False
                }
                episodes_list.append(ep_data)

        # School/Home/Parent assignments (licao de classe): real HAR shows
        # assignment_type=1 and activity_context="1" (NOT NewContext=13).
        _school_ctx = {"School": "1", "Home": "2", "Parent": "3"}
        for key in ["School", "Home", "Parent"]:
            for ass in init_data.get("Assignments", {}).get(key, []):
                episode_id = ass.get("Id") or ass.get("EpisodeId")
                pe = played_episodes.get(episode_id)
                slug = ass.get("Slug") or ass.get("EpisodeSlug")
                title = translate_matific_slug(slug)
                ep_data = {
                    "slug": slug,
                    "assignment_id": ass.get("AssignmentId"),
                    "campaign_id": ass.get("CampaignId") or default_campaign_id,
                    "context_id": 1,
                    "activity_context": ass.get("ContextId", _school_ctx.get(key, "1")),
                    "assignment_type": 1,
                    "due_date": ass.get("DueDate"),
                    "source": f"Atribuição: {key}",
                    "episode_id": episode_id,
                    "title": title,
                    "subtitle": "",
                    "problem_count": 5,
                    "order": ass.get("Order") or 0,
                    "completed": pe is not None and (pe.get("was_passed") or pe.get("highest_score") is not None),
                    "highest_score": pe["highest_score"] if pe else None,
                    "was_passed": pe["was_passed"] if pe else False
                }
                # activity_context is always a plain string like "1" in scoring facts
                try:
                    ep_data["activity_context"] = str(int(ep_data["activity_context"]))
                except Exception:
                    ep_data["activity_context"] = _school_ctx.get(key, "1")
                ep_data["context_id"] = int(ep_data["activity_context"]) if str(ep_data["activity_context"]).isdigit() else 1
                episodes_list.append(ep_data)

        if username in _matific_session_cache:
            _matific_session_cache[username]["episodes_assigned"] = (time.time(), episodes_list)
        return {"success": True, "episodes": episodes_list, "source": source}
    finally:
        await client.__aexit__(None, None, None)

@app.post("/api/matific/complete")
async def start_matific_simulation(req: MatificCompleteRequest, user: dict = Depends(get_current_user)):
    episode = req.episode
    if not episode or not episode.get("slug"):
        raise HTTPException(status_code=400, detail="Detalhes do episódio inválidos")

    username = user["username"]
    batch_state = get_user_matific_batch_state(username)
    if batch_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Existe um lote do Matific em execução. Aguarde ou cancele o lote.")

    for j in _matific_jobs.values():
        if j["username"] == username and j["status"] in ("pending", "running"):
            raise HTTPException(status_code=400, detail="Já existe uma simulação em andamento para esta conta.")

    job_id = uuid.uuid4().hex
    _matific_jobs[job_id] = {
        "status": "pending",
        "episode_slug": episode.get("slug"),
        "episode_title": episode.get("title", episode.get("slug")),
        "logs": ["Simulação de episódio agendada."],
        "progress_percent": 0.0,
        "username": username
    }
    asyncio.create_task(_matific_worker_task(username, _snapshot_user_credentials(user), episode, req.target_accuracy or "realistic", job_id, timings=req.timings))
    return {"success": True, "job_id": job_id}

@app.get("/api/matific/job/{job_id}")
async def get_matific_job_status(job_id: str, user: dict = Depends(get_current_user)):
    job = _matific_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    if job["username"] != user["username"]:
        raise HTTPException(status_code=403, detail="Não autorizado")
    return job

@app.post("/api/matific/job/{job_id}/stop")
async def stop_matific_job(job_id: str, user: dict = Depends(get_current_user)):
    job = _matific_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    if job["username"] != user["username"]:
        raise HTTPException(status_code=403, detail="Não autorizado")
    job["status"] = "stopped"
    return {"success": True}

@app.post("/api/matific/batch")
async def start_matific_batch(req: MatificBatchRequest, user: dict = Depends(get_current_user)):
    if not req.episodes:
        raise HTTPException(status_code=400, detail="Lista de episódios vazia")

    username = user["username"]
    state = get_user_matific_batch_state(username)
    if state["status"] == "running":
        raise HTTPException(status_code=400, detail="Já existe um lote em execução")

    for j in _matific_jobs.values():
        if j["username"] == username and j["status"] in ("pending", "running"):
            raise HTTPException(status_code=400, detail="Existe uma simulação individual em execução. Aguarde a conclusão.")

    asyncio.create_task(_matific_batch_worker(
        username, _snapshot_user_credentials(user), req.episodes,
        req.min_time_per_task or 1.0, req.max_time_per_task or 3.0,
        req.min_wait_between or 0.2, req.max_wait_between or 1.0,
        req.target_accuracy or "realistic"
    ))
    return {"success": True}

@app.get("/api/matific/batch/status")
async def get_matific_batch_status(user: dict = Depends(get_current_user)):
    username = user["username"]
    state = get_user_matific_batch_state(username)
    response_data = state.copy()
    if state["status"] == "running" and state["current_job_id"]:
        job = _matific_jobs.get(state["current_job_id"])
        if job:
            response_data["current_progress_percent"] = job.get("progress_percent", 0.0)
            response_data["active_job_logs"] = job.get("logs", [])[-5:]
    return response_data

@app.post("/api/matific/batch/stop")
async def stop_matific_batch(user: dict = Depends(get_current_user)):
    username = user["username"]
    state = get_user_matific_batch_state(username)
    if state["status"] == "running":
        state["status"] = "stopped"
        state["logs"].append("Lote interrompido pelo usuário.")
        if state["current_job_id"]:
            job = _matific_jobs.get(state["current_job_id"])
            if job:
                job["status"] = "stopped"
                job["logs"].append("Cancelado pelo cancelamento do lote.")
    return {"success": True}

@app.get("/api/matific/state")
async def get_matific_state(user: dict = Depends(get_current_user)):
    username = user["username"]
    cached_session = _matific_session_cache.get(username)
    if cached_session and cached_session.get("state"):
        ts, cached_state = cached_session["state"]
        if time.time() - ts < _MATIFIC_STATE_TTL and cached_state:
            return {"success": True, "state": cached_state, "cached": True}

    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")

    try:
        init_data = getattr(client, "init_data", None)
        if not init_data:
            init_data = await client.get_init_data()
            if init_data and username in _matific_session_cache:
                _matific_session_cache[username]["init_data"] = init_data

        if not init_data:
            raise HTTPException(status_code=502, detail="Falha ao obter dados de inicialização do Matific")

        campaigns = init_data.get("Campaigns", [])
        campaign_id = campaigns[0].get("Id") if campaigns else None
        if not campaign_id:
            for key in ["School", "Home", "Parent"]:
                asses = init_data.get("Assignments", {}).get(key, [])
                if asses:
                    campaign_id = asses[0].get("CampaignId")
                    break

        if not campaign_id:
            campaign_id = "default_campaign"

        state = await client.fetch_inventory_and_customization(campaign_id)
        result_state = {
            "coins": state["coins"],
            "xp": state["xp"],
            "rank": state["rank"],
            "weekly_goal": state["weekly_goal"],
            "weekly_goal_target": state["weekly_goal_target"],
            "inventory": state["inventory"],
            "customization": state["customization"],
            "campaign_id": campaign_id
        }
        if username in _matific_session_cache:
            _matific_session_cache[username]["state"] = (time.time(), result_state)
        return {
            "success": True,
            "state": result_state
        }
    finally:
        await client.__aexit__(None, None, None)

@app.post("/api/matific/purchase")
async def purchase_matific_item(req: MatificPurchaseRequest, user: dict = Depends(get_current_user)):
    username = user["username"]
    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")
    try:
        success = await client.purchase_item(req.campaign_id, req.item_id, req.cost)
        if not success:
            raise HTTPException(status_code=400, detail="Falha ao comprar o item. Saldo insuficiente ou erro no servidor.")
        _invalidate_matific_episodes_cache(username)
        return {"success": True, "message": "Item comprado com sucesso!"}
    finally:
        await client.__aexit__(None, None, None)

@app.post("/api/matific/equip")
async def equip_matific_item(req: MatificEquipRequest, user: dict = Depends(get_current_user)):
    username = user["username"]
    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")
    try:
        success = await client.equip_item(req.campaign_id, req.part_name, req.item_id)
        if not success:
            raise HTTPException(status_code=400, detail="Falha ao equipar o item.")
        _invalidate_matific_episodes_cache(username)
        return {"success": True, "message": "Item equipado com sucesso!"}
    finally:
        await client.__aexit__(None, None, None)

@app.post("/api/matific/repair-customization")
async def repair_matific_customization(req: MatificRepairRequest, user: dict = Depends(get_current_user)):
    username = user["username"]
    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")
    try:
        res = await client.repair_customization(req.campaign_id)
        _invalidate_matific_episodes_cache(username)
        return res
    finally:
        await client.__aexit__(None, None, None)

@app.post("/api/matific/set_stats")
async def set_matific_stats(req: MatificSetStatsRequest, user: dict = Depends(get_current_user)):
    username = user["username"]
    client = await _get_matific_client(user)
    if not client:
        raise HTTPException(status_code=401, detail="Falha na autenticação do Matific via SSO")
    try:
        success = await client.set_stats(req.campaign_id, coins=req.coins, xp=req.xp, rank=req.rank)
        if not success:
            raise HTTPException(status_code=502, detail="Falha ao atualizar estatísticas no Matific.")
        _invalidate_matific_episodes_cache(username)
        return {"success": True, "message": "Estatísticas atualizadas com sucesso!"}
    finally:
        await client.__aexit__(None, None, None)


class LeiaSPReadRequest(BaseModel):
    book_id: Optional[int] = None
    pages_to_read: Optional[int] = 0
    min_time: Optional[int] = 20
    max_time: Optional[int] = 40
    auto_solve_quiz: Optional[bool] = True
    sequential: Optional[bool] = False

async def _refresh_sed_leiasp_tokens(user: dict) -> bool:
    """Re-login on SED with stored credentials and refresh token_sed/leiasp_jwt.

    Sessions created at /api/login expire; LeiaSP auth with a stale token_sed
    then fails with 401 and the whole flow looked "broken". Re-login fixes it.
    """
    try:
        ra = (user.get("ra") or "").strip().lstrip("0") or (user.get("ra") or "").strip()
        digito = (user.get("digito") or "").strip()
        uf = (user.get("uf") or "").strip().upper()
        password = user.get("password") or ""
        if not ra or not password:
            return False
        constructed_user = f"{ra}{digito}{uf}"
        fp = _get_browser_context()
        headers_login = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "pt-BR,pt;q=0.9",
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": SED_SUBSCRIPTION_KEY,
            "X-Product-Name": "SalaDoFuturo",
            "Origin": "https://saladofuturo.educacao.sp.gov.br",
            "Referer": "https://saladofuturo.educacao.sp.gov.br/",
            "User-Agent": fp["user-agent"],
            "sec-ch-ua": fp["sec-ch-ua"],
            "sec-ch-ua-mobile": fp["sec-ch-ua-mobile"],
            "sec-ch-ua-platform": fp["sec-ch-ua-platform"],
        }
        async with HttpCloakClient(timeout=20.0) as client:
            resp_sed = await client.post(
                SED_LOGIN_URL, json={"user": constructed_user, "senha": password}, headers=headers_login)
            if resp_sed.status_code != 200:
                return False
            token_sed = resp_sed.json().get("token")
            if not token_sed:
                return False
            user["token_sed"] = token_sed
            headers_valida = dict(headers_login)
            headers_valida["Authorization"] = f"Bearer {token_sed}"
            url_leiasp = "https://sedintegracoes.educacao.sp.gov.br/saladofuturobffapi/integracoes/Token?plataforma=LeiaSP%2B"
            resp_tok = await client.get(url_leiasp, headers=headers_valida)
            if resp_tok.status_code == 200:
                jwt = resp_tok.json().get("data")
                if jwt:
                    user["leiasp_jwt"] = jwt
            return True
    except Exception as e:
        logger.warning(f"[LeiaSP] SED refresh failed: {e}")
        return False


async def _get_leiasp_client(user: dict) -> LeiaSPClient:
    token_sed = user.get("token_sed")
    leiasp_jwt = user.get("leiasp_jwt")
    if not token_sed and not leiasp_jwt:
        raise HTTPException(status_code=401, detail="Sessão SED inválida ou expirada.")
    client = LeiaSPClient(token_sed=token_sed, leiasp_jwt=leiasp_jwt)
    try:
        await client.authenticate()
        # Persist refreshed JWT (LeiaSPClient may have fetched a new one via token_sed)
        if getattr(client, "leiasp_jwt", None):
            user["leiasp_jwt"] = client.leiasp_jwt
        return client
    except Exception as e:
        logger.warning(f"[LeiaSP] Auth with stored tokens failed, trying SED re-login: {e}")
        try:
            await client.client.aclose()
        except Exception:
            pass
        if await _refresh_sed_leiasp_tokens(user):
            client2 = LeiaSPClient(token_sed=user.get("token_sed"), leiasp_jwt=user.get("leiasp_jwt"))
            try:
                await client2.authenticate()
                user["leiasp_jwt"] = client2.leiasp_jwt
                return client2
            except Exception as e2:
                logger.error(f"[LeiaSP] Erro ao autenticar no Elefante Letrado após refresh: {e2}")
                raise HTTPException(status_code=502, detail=f"Erro ao autenticar no LeiaSP/Elefante Letrado: {e2}")
        logger.error(f"[LeiaSP] Erro ao autenticar no Elefante Letrado: {e}")
        raise HTTPException(status_code=502, detail=f"Erro ao autenticar no LeiaSP/Elefante Letrado: {e}")

async def _wait_job(job: dict, seconds: int = 0) -> bool:
    for _ in range(max(1, seconds)):
        while job.get("status") == "paused":
            await asyncio.sleep(0.5)
            if job.get("status") == "stopped":
                return False
        if job.get("status") == "stopped":
            return False
        if seconds > 0:
            await asyncio.sleep(1)
    return job.get("status") != "stopped"

async def _read_single_leiasp_book_core(
    client: LeiaSPClient,
    book_id: int,
    pages_to_read: int,
    min_time: int,
    max_time: int,
    auto_solve_quiz: bool,
    job: dict
) -> bool:
    job["logs"].append(f"Obtendo metadados do livro ID {book_id}...")
    try:
        meta = await client.get_book_metadata(book_id)
    except Exception as e:
        job["logs"].append(f"Erro ao buscar metadados do livro {book_id}: {e}")
        return False

    book_title = meta.get("BookTitle") or meta.get("Title") or f"Livro #{book_id}"
    cover_path = meta.get("CoverPageUrl") or meta.get("ThumbnailCoverPic") or meta.get("CoverUrl") or meta.get("CoverThumbnailUrl") or meta.get("UrlToCoverImage") or ""
    if cover_path.startswith("/"):
        cover_path = f"{client.ELEFANTE_CDN_BASE}{cover_path}"

    job["book_id"] = book_id
    job["book_title"] = book_title
    job["book_cover_url"] = cover_path

    book_context = {
        "title": book_title,
        "authors": meta.get("Authors") or meta.get("Author") or "",
        "publisher": meta.get("Publisher") or "",
        "synopsis": meta.get("Synopsis") or meta.get("Description") or "",
        "isbn": meta.get("Isbn") or "",
    }

    is_quiz_active = bool(meta.get("IsQuizActive"))
    # NumberPages is absent from /student/books/{id} metadata; the library
    # catalog (/library/discover + /readings) carries the real total. Fall back
    # to it before defaulting to 100, otherwise CFI map and progress use a
    # wrong PageCount and the API ignores the records.
    total_pages = meta.get("NumberPages") or meta.get("TotalPages") or 0
    if not total_pages:
        try:
            for b in await client.get_library_books():
                if int(b.get("id") or 0) == int(book_id) and int(b.get("total_pages") or 0) > 0:
                    total_pages = int(b["total_pages"])
                    break
        except Exception:
            pass
    if not total_pages:
        total_pages = 100
    try:
        total_pages = int(total_pages)
    except Exception:
        total_pages = 100
    current_page = 1

    reading_info = meta.get("Reading")
    if isinstance(reading_info, list):
        for r in reading_info:
            if r.get("Type") == "Read":
                current_page = r.get("Page") or 1
                break

    epub_url_path = meta.get("UrlToEpubFile") or meta.get("FullUrlToEpubFile") or ""
    if epub_url_path.startswith("http"):
        epub_url = epub_url_path
    else:
        epub_url = f"{client.ELEFANTE_CDN_BASE}{epub_url_path}" if epub_url_path.startswith("/") else epub_url_path

    page_cfi_map = []
    epub_hash = ""
    if epub_url:
        job["logs"].append("Baixando EPUB para mapear spine e calcular CFIs Colibrio...")
        epub_hash, page_cfi_map = await client.parse_epub(epub_url, total_pages)
        if page_cfi_map:
            job["logs"].append(f"Mapa CFI construído com sucesso: {len(page_cfi_map)} páginas mapeadas.")

    job["total_pages"] = total_pages
    job["current_page"] = current_page
    job["logs"].append(f"Iniciando: '{book_title}' | Páginas: {total_pages} | Página atual: {current_page}")

    pages_to_read_actual = pages_to_read if pages_to_read > 0 else (total_pages - current_page + 1)
    target_page = min(total_pages, current_page + pages_to_read_actual - 1)

    if current_page <= target_page and current_page < total_pages:
        if epub_url:
            await client.simulate_colibrio_epub_load(epub_url, job)
        await client.start_session(book_id)

        pages_read_count = 0
        for page in range(current_page, target_page + 1):
            read_time = random.randint(min_time, max_time)
            job["logs"].append(f"Lendo página {page}/{total_pages} por {read_time}s...")

            if not await _wait_job(job, read_time):
                job["logs"].append("Leitura interrompida pelo usuário.")
                return False

            if page_cfi_map and page <= len(page_cfi_map):
                cfi_value = page_cfi_map[page - 1]
            else:
                spine_item = (page * 2) + 2
                h = epub_hash or "0000000000000000000000000000000000000000"
                cfi_value = f"com.colibrio.epub.signature:{h}#epubcfi(/6/{spine_item}!/4/1:0)"

            is_final_page = (page == target_page and page >= total_pages)
            try:
                if is_final_page:
                    finish_res = await client.finish_book(book_id, total_pages, cfi_value, read_time)
                    prog = finish_res.get("progress", {})
                    if prog.get("success"):
                        is_ok = finish_res.get("is_completed_with_success")
                        pts = finish_res.get("points", 0)
                        job["logs"].append(f"Livro finalizado na API LeiaSP! {'(Sucesso total)' if is_ok else '(Tempo registrado)'} | Pontos: {pts}")
                    else:
                        job["logs"].append(f"Aviso ao finalizar livro: {prog.get('error', 'Sem resposta')}")
                else:
                    prog_res = await client.send_page_progress(book_id, page, total_pages, cfi_value, read_time)
                    if prog_res.get("success"):
                        job["logs"].append(f"Página {page}/{total_pages} registrada com sucesso.")
                    else:
                        job["logs"].append(f"Aviso no registro da página {page}: {prog_res.get('error', 'Falha')}")
            except Exception as e:
                job["logs"].append(f"Erro ao registrar página {page}: {e}")

            pages_read_count += 1
            job["current_page"] = page
            job["progress_percent"] = round((pages_read_count / max(pages_to_read_actual, 1)) * 100, 1)

            if not is_final_page and ((pages_read_count % 10 == 0) or (page == target_page)):
                try:
                    close_res = await client.close_book_checkpoint(book_id, read_time)
                    if close_res:
                        job["logs"].append(f"Checkpoint intermediário salvo (pág {page})")
                except Exception as e:
                    job["logs"].append(f"Aviso no checkpoint: {e}")

    if auto_solve_quiz and is_quiz_active and job.get("status") != "stopped":
        job["logs"].append("Verificando questionário do livro...")
        try:
            quiz_data = await client.fetch_quiz(book_id)
            if quiz_data.get("QuizEnabled") and not quiz_data.get("QuizComplete"):
                questions = quiz_data.get("Question", [])
                if questions:
                    job["logs"].append(f"Quiz ativo com {len(questions)} questões. Resolvendo...")
                    q_answers = {}
                    for idx, q in enumerate(questions):
                        if job.get("status") == "stopped":
                            break
                        q_id = q.get("Id")
                        q_type = q.get("QuestionTypeId")
                        q_text = q.get("Text", "")

                        cached = get_cached_leiasp_quiz_answer(book_id, q_id)
                        if cached:
                            job["logs"].append(f"Questão {idx+1}: resposta do cache SQLite.")
                            if q_type != 10:
                                q_answers[q_id] = {"Id": int(q_id), "Answers": cached if isinstance(cached, list) else [cached], "TrueFalseCorrect": True}
                            else:
                                q_answers[q_id] = {
                                    "Id": int(q_id), "Answers": [], "TrueFalseCorrect": True,
                                    "DissertativeResponse": str(cached), "DissertativeEvaluation": "correta",
                                    "DissertativeEvaluationResult": "correta", "DissertativeStudentFeedback": "",
                                    "DissertativeJsonTextFormEvaluation": "{}", "DissertativeElephantTips": ""
                                }
                            continue

                        if q_type != 10:
                            correct_ids = [int(ans.get("Id")) for ans in q.get("Answer", []) if ans.get("IsCorrectAnswer")]
                            if not correct_ids:
                                job["logs"].append(f"Questão {idx+1}: consultando IA...")
                                correct_ids = await resolve_leiasp_objective(q, book_context)
                                if not correct_ids and q.get("Answer"):
                                    correct_ids = [int(q["Answer"][0].get("Id"))]
                            save_cached_leiasp_quiz_answer(book_id, q_id, q_text, "objective", q.get("Answer", []), correct_ids)
                            q_answers[q_id] = {"Id": int(q_id), "Answers": correct_ids, "TrueFalseCorrect": True}
                        else:
                            job["logs"].append(f"Questão {idx+1} (Dissertativa): gerando resposta com IA...")
                            student_resp = await resolve_leiasp_dissertative(q_text, book_context)
                            eval_data = {}
                            try:
                                eval_data = await client.evaluate_dissertative_answer(book_id, int(q_id), q_text, student_resp)
                            except Exception:
                                pass
                            save_cached_leiasp_quiz_answer(book_id, q_id, q_text, "dissertative", [], student_resp)
                            q_answers[q_id] = {
                                "Id": int(q_id), "Answers": [], "TrueFalseCorrect": True,
                                "DissertativeResponse": student_resp,
                                "DissertativeEvaluation": eval_data.get("EvaluationResult") or "correta",
                                "DissertativeEvaluationResult": eval_data.get("EvaluationResult") or "correta",
                                "DissertativeStudentFeedback": eval_data.get("StudentFeedback") or "",
                                "DissertativeJsonTextFormEvaluation": eval_data.get("JsonTextFormEvaluation") or "{}",
                                "DissertativeElephantTips": eval_data.get("ElephantTips") or ""
                            }

                    by_milestone = {}
                    for q in questions:
                        by_milestone.setdefault(q.get("MilestonePercentage"), []).append(q)

                    for m in sorted(by_milestone.keys(), key=lambda x: (x is None, x or 0)):
                        if job.get("status") == "stopped":
                            break
                        g_answers = [q_answers[q.get("Id")] for q in by_milestone[m] if q.get("Id") in q_answers]
                        if g_answers:
                            res_m = await client.submit_milestone_quiz(book_id, m, g_answers)
                            approved = res_m.get("Approved", False) if isinstance(res_m, dict) else False
                            job["logs"].append(f"Milestone {m}% enviado | Aprovado={approved}")

                    if job.get("status") != "stopped":
                        await client.finish_quiz(book_id)
                        job["logs"].append("Quiz do livro concluído!")
            else:
                job["logs"].append("Nenhum quiz pendente para este livro.")
        except Exception as e:
            job["logs"].append(f"Aviso durante verificação de quiz: {e}")

    job["logs"].append(f"Livro '{book_title}' finalizado com sucesso!")
    return True

async def _leiasp_reader_worker(
    session_data: dict,
    book_id: Optional[int],
    pages_to_read: int,
    min_time: int,
    max_time: int,
    auto_solve_quiz: bool,
    sequential: bool,
    job_id: str
):
    job = _leiasp_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    job["logs"].append("Iniciando sessão do LeiaSP...")

    client = LeiaSPClient(token_sed=session_data.get("token_sed"), leiasp_jwt=session_data.get("leiasp_jwt"))
    try:
        await client.authenticate()
        job["logs"].append("Autenticado com sucesso no Elefante Letrado!")

        if not sequential and book_id:
            await _read_single_leiasp_book_core(
                client=client, book_id=book_id, pages_to_read=pages_to_read,
                min_time=min_time, max_time=max_time, auto_solve_quiz=auto_solve_quiz, job=job
            )
        else:
            job["logs"].append("Modo Sequencial: obtendo catálogo de livros incompletos...")
            books = await client.get_library_books()
            incomplete = [b for b in books if not b.get("is_complete") and b.get("progress", 0) < 100.0]

            if not incomplete:
                job["logs"].append("Nenhum livro incompleto encontrado na biblioteca.")
            else:
                job["queue_total"] = len(incomplete)
                job["queue_completed"] = []
                job["logs"].append(f"Encontrados {len(incomplete)} livros incompletos. Iniciando fila...")
                for idx, b in enumerate(incomplete):
                    if not await _wait_job(job):
                        job["logs"].append("Fila sequencial interrompida pelo usuário.")
                        break

                    job["queue_current_idx"] = idx + 1
                    job["queue_current_title"] = b.get("title")
                    job["logs"].append(f"\n--- [Livro {idx+1}/{len(incomplete)}] {b.get('title')} ---")
                    await _read_single_leiasp_book_core(
                        client=client, book_id=b["id"], pages_to_read=0,
                        min_time=min_time, max_time=max_time, auto_solve_quiz=auto_solve_quiz, job=job
                    )
                    job["queue_completed"].append(b.get("title"))

                    if idx + 1 < len(incomplete) and job["status"] not in ("stopped", "paused"):
                        pause_sec = random.randint(15, 30)
                        job["logs"].append(f"Pausa humanizada de {pause_sec}s entre livros...")
                        if not await _wait_job(job, pause_sec):
                            break

        if job["status"] not in ("stopped", "failed"):
            job["status"] = "completed"
            job["progress_percent"] = 100.0
            job["logs"].append("Todas as operações do LeiaSP foram concluídas com sucesso!")

    except Exception as e:
        logger.error(f"[LeiaSP Worker] Erro no worker: {e}", exc_info=True)
        job["status"] = "failed"
        job["logs"].append(f"Erro durante a execução: {e}")
    finally:
        await client.client.aclose()

@app.get("/api/leiasp/books")
async def get_leiasp_books_endpoint(user: dict = Depends(get_current_user)):
    client = await _get_leiasp_client(user)
    try:
        books = await client.get_library_books()
        return {"success": True, "books": books, "total": len(books)}
    finally:
        await client.client.aclose()

@app.get("/api/leiasp/active-job")
async def get_active_leiasp_job_endpoint(user: dict = Depends(get_current_user)):
    username = user["username"]
    user_jobs = [j for j in _leiasp_jobs.values() if j.get("username") == username]
    if not user_jobs:
        return {"active": False, "job": None}

    user_jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    active = [j for j in user_jobs if j.get("status") in ("running", "paused", "pending")]
    if active:
        return {"active": True, "job": active[0]}

    recent = user_jobs[0]
    if time.time() - recent.get("created_at", 0) < 45:
        return {"active": True, "job": recent}

    return {"active": False, "job": None}

@app.post("/api/leiasp/read")
async def start_leiasp_reading_endpoint(req: LeiaSPReadRequest, user: dict = Depends(get_current_user)):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "username": user["username"],
        "book_id": req.book_id,
        "book_title": "Carregando...",
        "book_cover_url": "",
        "sequential": req.sequential,
        "queue_total": 0,
        "queue_current_idx": 0,
        "queue_current_title": "",
        "queue_completed": [],
        "status": "pending",
        "progress_percent": 0.0,
        "current_page": 0,
        "total_pages": 0,
        "logs": ["Job criado. Aguardando inicialização..."],
        "created_at": time.time(),
    }
    _leiasp_jobs[job_id] = job

    min_t = max(10, req.min_time or 20)
    max_t = max(min_t, req.max_time or 40)

    asyncio.create_task(
        _leiasp_reader_worker(
            session_data=_snapshot_user_credentials(user),
            book_id=req.book_id,
            pages_to_read=req.pages_to_read or 0,
            min_time=min_t,
            max_time=max_t,
            auto_solve_quiz=bool(req.auto_solve_quiz),
            sequential=bool(req.sequential),
            job_id=job_id
        )
    )

    return {"success": True, "job_id": job_id, "message": "Leitura iniciada com sucesso em background."}

@app.get("/api/leiasp/job/{job_id}")
async def get_leiasp_job_status_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    job = _require_job_owner(_leiasp_jobs.get(job_id), user, "Job de leitura")
    return {"success": True, "job": job}

@app.post("/api/leiasp/job/{job_id}/pause")
async def pause_leiasp_job_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    job = _require_job_owner(_leiasp_jobs.get(job_id), user, "Job de leitura")
    if job.get("status") == "running":
        job["status"] = "paused"
        job["logs"].append("⏸️ Leitura pausada pelo usuário.")
    return {"success": True, "message": "Job pausado com sucesso.", "job": job}

@app.post("/api/leiasp/job/{job_id}/resume")
async def resume_leiasp_job_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    job = _require_job_owner(_leiasp_jobs.get(job_id), user, "Job de leitura")
    if job.get("status") == "paused":
        job["status"] = "running"
        job["logs"].append("▶️ Leitura retomada pelo usuário.")
    return {"success": True, "message": "Job retomado com sucesso.", "job": job}

@app.post("/api/leiasp/job/{job_id}/stop")
async def stop_leiasp_job_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    job = _require_job_owner(_leiasp_jobs.get(job_id), user, "Job de leitura")
    job["status"] = "stopped"
    job["logs"].append("⏹️ Comando de parada recebido do usuário.")
    return {"success": True, "message": "Job interrompido com sucesso.", "job": job}


# Hyper Futuro X | consultas oficiais de leitura, com lista fechada de rotas.
# Não utiliza cookies, tokens ou códigos de alunos copiados de terceiros.
_HYPER_SED_BASE = "https://sedintegracoes.educacao.sp.gov.br/saladofuturobffapi"
_HYPER_API_ROUTES = {
    "perfil": ("alunoapi/api/Aluno/ExibirAluno", "ra"),
    "informacoes": ("ncaapi/api/TransferenciaSF/ObterInformacaoAluno", "student"),
    "boletim": ("apiboletim/api/Fechamento/ConsultaFechamentoComparativo", "student"),
    "faltas": ("apiboletim/api/Frequencia/GetFaltasBimestreAtual", "student"),
    "agenda": ("apiboletim/api/Agenda/GetAgendaPeriodoEscola", "student"),
    "disciplinas": ("apihubintegracoes/api/v2/Disciplina/ListarDisciplinaPorAluno", "student"),
    "notificacoes": ("cmspwebservice/api/sala-do-futuro-alunos/consulta-notificacao-cmsp", "user"),
    "avatar": ("avatarapi/api/usuarios/obter-configuracao-avatar", "user"),
    "avisos": ("muralavisosapi/api/mural-avisos/listar-avisos-turma", "class"),
    "conteudos": ("muralavisosapi/api/conteudo-digital/listar-qtde-subcategoria", "school"),
    "conquistas": ("apisalaconquistas/api/salaConquista/conquistaAluno", "user"),
    "monitor": ("muralavisosapi/api/aluno-monitor/aluno-monitor-valido", "student"),
    "parametros": ("ncaapi/api/ItinerarioFormativoSF/ListarParametros", "none"),
    "configuracoes": ("api/configuracoes", "none"),
}

def _hyper_api_params(kind: str, user: dict):
    from datetime import date, timedelta
    student_code = user.get("student_id")
    user_code = user.get("sed_user_code")
    if kind == "ra":
        ra = str(user.get("ra") or "")
        if not ra.isdecimal() or not user.get("digito"):
            return None, "RA e dígito indisponíveis na sessão."
        return {"inNumRA": ra.zfill(12), "inDigitoRA": str(user["digito"]), "inSiglaUFRA": "SP"}, None
    if kind in ("student", "user", "class", "school"):
        code = student_code if kind == "student" else user_code
        if not code or not str(code).isascii() or not str(code).isdecimal():
            return None, "O login não retornou o código escolar exigido por esta consulta. Verifique a fonte do código no fluxo oficial; nenhum ID foi presumido."
        if kind == "student":
            return {"codigoAluno": code}, None
        if kind == "user":
            return {"UsuarioId": code}, None
        # A turma é obtida somente do contexto autenticado; nunca de números copiados do Network.
        if kind == "class":
            class_code = str(user.get("school_class_code") or "")
            if class_code.isascii() and class_code.isdecimal() and user_code:
                return {"CodigoUsuario": user_code, "PerfilAviso": 1, "Turmas": class_code}, None
            return None, "Código da turma ainda não identificado na sessão. Consulte Informações escolares primeiro."
        return None, "Parâmetros escolares ainda não disponíveis na sessão."
    return {}, None

def _hyper_sanitize(obj, depth=0):
    """Remove campos pessoais e credenciais das respostas visualizadas no Hyper."""
    if depth > 7:
        return "…"
    blocked = ("cpf", "endereco", "logradouro", "telefone", "celular", "nascimento", "documento", "token", "password", "senha", "cookie", "authorization", "secret", "chave", "email", "cep", "requestid", "operador", "affinity", "outnumra", "outdigitora", "outcdmatricula", "nomeusuario", "nomealuno", "outnomealuno")
    if isinstance(obj, dict):
        out = {}
        for key, value in list(obj.items())[:80]:
            if any(b in str(key).casefold() for b in blocked): continue
            out[str(key)[:90]] = _hyper_sanitize(value, depth + 1)
        return out
    if isinstance(obj, list): return [_hyper_sanitize(x, depth+1) for x in obj[:60]]
    if isinstance(obj, str): return obj[:300]
    if isinstance(obj, (int, float, bool)) or obj is None: return obj
    return str(obj)[:100]

@app.get("/api/hyper/escolar/catalogo")
async def hyper_escolar_catalogo(user: dict = Depends(get_current_user)):
    return {"apis": list(_HYPER_API_ROUTES), "aluno_id_disponivel": bool(user.get("student_id")), "usuario_id_disponivel": bool(user.get("sed_user_code")), "observacao": "Dados obtidos apenas com a sessão autenticada; disponibilidade depende dos serviços externos."}

# Integração opcional NCA009 (FormacaoClasse): requer credencial PRÓPRIA,
# concedida pelo serviço de integração SED. O token de login do aluno NÃO é
# equivalente ao token NCA009 gerado pelo ValidarUsuario.
# Nunca devolver ou registrar a lista outAlunos (dados de outros estudantes).
_HYPER_NCA_FORMACAO_URL = "https://integracaosed.educacao.sp.gov.br/ncaapi/api/RelacaoAlunosClasse/FormacaoClasse"


def _hyper_formacao_summary(payload, user):
    """Aceita somente a classe com vínculo confirmado à sessão.

    Não registra ou propaga a lista privada de alunos da resposta NCA009.
    """
    if not isinstance(payload, dict) or payload.get("outErro"):
        return None
    class_code = str(user.get("school_class_code") or "")
    response_class = str(payload.get("outNumClasse") or "")
    if not class_code or response_class != class_code:
        return None
    expected_school = str(user.get("school_code") or "")
    response_school = str(payload.get("outCodEscola") or "")
    if expected_school and response_school != expected_school:
        return None
    # A consulta de formação pode trazer alunos de várias situações de matrícula.
    # Apenas confirmar a própria identidade, sem expor o restante da lista.
    own_ra = str(user.get("ra") or "").lstrip("0") or "0"
    own_digit = str(user.get("digito") or "")
    roster = payload.get("outAlunos")
    if not isinstance(roster, list) or not any(
        isinstance(member, dict)
        and (str(member.get("outNumRA") or "").lstrip("0") or "0") == own_ra
        and str(member.get("outDigitoRA") or "") == own_digit
        and str(member.get("outSiglaUFRA") or "").upper() == "SP"
        for member in roster
    ):
        return None
    name = payload.get("outDescNomeAbrevEscola")
    group = payload.get("outTurma")
    shift = payload.get("outDescricaoTurno")
    series = payload.get("outCodSerieAno")
    # Não incorporar campos nulos, numéricos ou grandes como nomes da escola.
    school_name = name.strip()[:120] if isinstance(name, str) and 3 <= len(name.strip()) <= 120 and not name.strip().isdecimal() else None
    group_label = group.strip()[:12] if isinstance(group, str) and 1 <= len(group.strip()) <= 12 and not group.strip().isdecimal() else None
    shift_label = shift.strip()[:45] if isinstance(shift, str) and 2 <= len(shift.strip()) <= 45 else None
    series_label = series.strip()[:12] if isinstance(series, (str, int)) and str(series).strip() else None
    class_name = " · ".join(part for part in (
        (f"{series_label}ª série" if series_label and series_label.isdecimal() and int(series_label) in range(1, 10) else None),
        (f"Turma {group_label}" if group_label else None), shift_label,
    ) if part) or None
    return {"school": school_name, "class_name": class_name}


@app.get("/api/hyper/escolar/resolver-turma")
async def hyper_resolver_turma(user: dict = Depends(get_current_user)):
    """Resolve nomes oficiais somente para a matrícula autenticada e autorizada.

    A API NCA009 exige token independente; não reutilizar o JWT do aluno.
    """
    import httpx
    token = os.environ.get("SED_NCA_FORMACAO_TOKEN", "").strip()
    if not token:
        return {"ok": False, "status": "configuracao", "message": "Consulta de nomes da turma ainda não habilitada na integração SED."}
    class_code = str(user.get("school_class_code") or "")
    if not class_code.isascii() or not class_code.isdecimal() or not (1 <= len(class_code) <= 9):
        return {"ok": False, "status": "turma_pendente", "message": "Código da turma ainda não confirmado na sessão."}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=False) as client:
            response = await client.get(_HYPER_NCA_FORMACAO_URL,
                                        params={"inNumClasse": class_code}, headers=headers)
    except httpx.RequestError:
        return {"ok": False, "status": "conexao", "message": "Não foi possível consultar a formação da turma."}
    if response.status_code != 200:
        return {"ok": False, "status": "servico", "message": "A integração SED não autorizou ou não completou a consulta."}
    if len(response.content) > 2 * 1024 * 1024:
        return {"ok": False, "status": "limite", "message": "Resposta da integração acima do limite permitido."}
    try:
        payload = response.json()
    except ValueError:
        return {"ok": False, "status": "formato", "message": "Resposta da integração inválida."}
    summary = _hyper_formacao_summary(payload, user)
    # Descartar resposta que inclui terceiros; não transmitir payload ao navegador.
    del payload
    if summary is None:
        return {"ok": False, "status": "vinculo", "message": "Não foi possível confirmar a turma da matrícula autenticada."}
    if summary["school"]:
        user["school_name"] = summary["school"]
    if summary["class_name"]:
        user["school_class_name"] = summary["class_name"]
    return {"ok": True, "school": user.get("school_name"), "class_name": user.get("school_class_name"), "source": "NCA009"}


@app.get("/api/hyper/escolar/diagnostico-identificadores")
async def hyper_diagnostico_identificadores(user: dict = Depends(get_current_user)):
    """Diagnóstico da sessão atual: nunca devolve IDs, RA, tokens ou senhas."""
    return {
        "sessao_autenticada": True,
        "codigo_aluno_encontrado": bool(user.get("student_id")),
        "codigo_usuario_encontrado": bool(user.get("sed_user_code")),
        "campos_login": user.get("login_field_names", []),
        "campos_dados_usuario": user.get("login_user_field_names", []),
        "campos_token": user.get("login_claim_field_names", []),
        "observacao": (
            "Código de aluno identificado explicitamente na autenticação."
            if user.get("student_id") else
            "A autenticação não forneceu campo de código de aluno confirmado; "
            "CD_USUARIO não será usado como AlunoId sem validação."
        ),
    }

async def _hyper_consultar_fechamentos(user: dict):
    """Lê fechamentos reais de 2026, sem calcular médias ou preencher bimestres vazios."""
    import httpx
    from datetime import date
    code = user.get("student_id")
    if not code: return {"ok": False, "status": "parametros_ausentes", "message": "Código do aluno ausente. Vincule sua matrícula nesta sessão."}
    token = user.get("token_sed")
    if not token: return {"ok": False, "message": "Sessão SED indisponível."}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "X-Product-Name": "SalaDoFuturo",
        "Origin": "https://saladofuturo.educacao.sp.gov.br", "Referer": "https://saladofuturo.educacao.sp.gov.br/"}
    if SED_SUBSCRIPTION_KEY: headers["Ocp-Apim-Subscription-Key"] = SED_SUBSCRIPTION_KEY
    url = f"{_HYPER_SED_BASE}/apiboletim/api/Fechamento/ConsultaFechamentoComparativo"
    periods = {5: 1, 6: 2, 7: 3, 8: 4}
    result = []
    available = []
    errors = []
    async with httpx.AsyncClient(timeout=14, follow_redirects=False) as client:
        for fechamento_id, bim in periods.items():
            try:
                r = await client.get(url, params={"codigoAluno": code, "anoLetivo": date.today().year,
                    "somenteAtivo": 0, "tipoFechamento": fechamento_id, "codigoDisciplina": 0}, headers=headers)
                if r.status_code != 200 or len(r.content) > 1024*1024:
                    errors.append(bim); continue
                payload = r.json()
                rows = payload.get("data") if isinstance(payload, dict) and payload.get("isSucess") is True else None
                if not isinstance(rows, list): errors.append(bim); continue
                available.append(bim)
                for item in rows[:100]:
                    if not isinstance(item, dict) or item.get("tipoFechamentoId") != fechamento_id or str(item.get("alunoId")) != str(code): continue
                    turma = item.get("turmaId")
                    escola = item.get("escolaId")
                    if isinstance(turma,int) and turma>0: user["school_class_code"] = str(turma)
                    if isinstance(escola,int) and escola>0: user["school_code"] = str(escola)
                    result.append({"disciplinaId": item.get("disciplinaId"),
                        "nomeDisciplina": str(item.get("nomeDisciplina") or "Disciplina").strip()[:100],
                        "bimestre": bim, "notaAtribuida": item.get("notaAtribuida") if isinstance(item.get("notaAtribuida"),(int,float)) else None,
                        "notaMediaTurma": item.get("notaMediaTurma") if isinstance(item.get("notaMediaTurma"),(int,float)) else None})
            except (httpx.RequestError, ValueError): errors.append(bim)
    if not available:
        return {"ok": False, "message": "Não foi possível consultar nenhum fechamento do boletim.", "periodos_com_erro": errors}
    return {"ok": True, "recurso": "boletim", "dados": {"data": result, "periodos_consultados": available,
        "periodos_com_erro": errors, "anoLetivo": date.today().year}, "status": "recebido"}

async def _hyper_link_verified_student(user: dict, candidate: str):
    """Vincula apenas AlunoId confirmado pelo boletim da própria matrícula/RA da sessão."""
    import httpx
    from datetime import date
    if not isinstance(candidate, str) or not (candidate.isascii() and candidate.isdecimal()) or not (1 <= len(candidate) <= 14):
        return False, "Código do aluno inválido."
    token = user.get("token_sed")
    if not token:
        return False, "Sessão SED indisponível."
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "X-Product-Name": "SalaDoFuturo",
        "Origin": "https://saladofuturo.educacao.sp.gov.br", "Referer": "https://saladofuturo.educacao.sp.gov.br/"}
    if SED_SUBSCRIPTION_KEY: headers["Ocp-Apim-Subscription-Key"] = SED_SUBSCRIPTION_KEY
    url = f"{_HYPER_SED_BASE}/apiboletim/api/Fechamento/ConsultaFechamentoComparativo"
    try:
        async with httpx.AsyncClient(timeout=14, follow_redirects=False) as client:
            r = await client.get(url, params={"codigoAluno": candidate, "anoLetivo": date.today().year,
                "somenteAtivo": 0, "tipoFechamento": 5, "codigoDisciplina": 0}, headers=headers)
    except httpx.RequestError:
        return False, "Não foi possível validar com a SED."
    if r.status_code != 200 or len(r.content) > 1024*1024:
        return False, "A SED não autorizou a validação do código."
    try: raw = r.json()
    except ValueError: return False, "Resposta inválida da SED."
    rows = raw.get("data") if isinstance(raw, dict) and raw.get("isSucess") is True else None
    if not isinstance(rows, list) or not rows:
        return False, "Não há fechamento com RA disponível para validar a matrícula."
    own_ra = str(user.get("ra") or "").lstrip("0") or "0"
    own_digit = str(user.get("digito") or "")
    if not all(isinstance(row, dict) and str(row.get("alunoId")) == candidate
        and (str(row.get("numeroRa") or "").lstrip("0") or "0") == own_ra
        and str(row.get("numeroDigitoRa") or "") == own_digit for row in rows):
        return False, "O código informado não corresponde ao RA da sessão autenticada."
    user["student_id"] = candidate
    for row in rows:
        cls = row.get("turmaId")
        school = row.get("escolaId")
        if isinstance(cls, int) and cls > 0: user["school_class_code"] = str(cls)
        if isinstance(school, int) and school > 0: user["school_code"] = str(school)
    return True, "Matrícula validada na SED para esta sessão."

class HyperStudentLink(BaseModel):
    codigo_aluno: str

@app.post("/api/hyper/escolar/vincular-aluno")
async def hyper_vincular_aluno(payload: HyperStudentLink, user: dict = Depends(get_current_user)):
    ok, message = await _hyper_link_verified_student(user, payload.codigo_aluno.strip())
    return {"ok": ok, "message": message, "student_code_available": bool(user.get("student_id")),
        "class_code_available": bool(user.get("school_class_code"))}

@app.get("/api/hyper/escolar/{recurso}")
async def hyper_escolar_recurso(recurso: str, user: dict = Depends(get_current_user)):
    from datetime import date, timedelta
    import httpx
    route = _HYPER_API_ROUTES.get(recurso)
    if not route:
        raise HTTPException(404, "Consulta não cadastrada")
    path, kind = route
    params, error = _hyper_api_params(kind, user)
    if error:
        return {"ok": False, "status": "parametros_ausentes", "message": error}
    if recurso == "boletim":
        # Os fechamentos 5/6/7/8 foram identificados no Network do boletim.
        # O tipo 10 não é interpretado como média final sem confirmação do seu formato.
        return await _hyper_consultar_fechamentos(user)
    elif recurso == "informacoes": params = {"cdAluno": user["student_id"], "anoLetivo": date.today().year}
    elif recurso == "agenda":
        now = date.today()
        params.update({"anoLetivo": now.year, "dataInicio": now.isoformat(), "dataFim": (now + timedelta(days=30)).isoformat()})
    elif recurso == "avatar": params = {"idUsuario": user["sed_user_code"]}
    elif recurso == "notificacoes": params = {"userId": user["sed_user_code"]}
    elif recurso == "conquistas": params = {"CodigoAluno": user["sed_user_code"]}
    token = user.get("token_sed")
    if not token: raise HTTPException(401, "Sessão SED indisponível")
    headers = {
        "Authorization": f"Bearer {token}", "Accept": "application/json",
        "Origin": "https://saladofuturo.educacao.sp.gov.br",
        "Referer": "https://saladofuturo.educacao.sp.gov.br/",
        "X-Product-Name": "SalaDoFuturo",
    }
    if SED_SUBSCRIPTION_KEY:
        headers["Ocp-Apim-Subscription-Key"] = SED_SUBSCRIPTION_KEY
    url = f"{_HYPER_SED_BASE}/{path}"
    try:
        async with httpx.AsyncClient(timeout=13, follow_redirects=False) as client:
            r = await client.get(url, params=params, headers=headers)
    except httpx.RequestError:
        return {"ok": False, "status": "conexao", "message": "Não foi possível conectar ao serviço escolar."}
    if r.status_code != 200:
        return {"ok": False, "status": "http", "http_status": r.status_code, "message": "O serviço não autorizou ou não concluiu a consulta."}
    if len(r.content) > 1024 * 1024:
        return {"ok": False, "status": "limite", "message": "Resposta grande demais para exibir."}
    try:
        result = r.json()
    except ValueError:
        return {"ok": False, "status": "formato", "message": "A resposta não veio em JSON."}
    if recurso == "perfil":
        details = result.get("data", {}).get("outDadosPessoais", {}) if isinstance(result, dict) else {}
        if isinstance(details, dict) and details.get("outNomeAluno"):
            user["name"] = str(details["outNomeAluno"])[:90]
    if recurso == "informacoes":
        # Extrair campos escolares apenas quando vierem rotulados explicitamente na resposta.
        # Não deduzir turma a partir de identificadores de outras APIs.
        possible = ("codigoTurma", "CodigoTurma", "cdTurma", "CdTurma", "idTurma", "IdTurma")
        school_keys = ("nomeEscola", "NomeEscola", "nmEscola", "NmEscola", "escola")
        class_keys = ("nomeTurma", "NomeTurma", "nmTurma", "NmTurma", "turmaDescricao")
        def seek_labels(obj, keys, depth=0):
            if depth > 6: return None
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj and isinstance(obj[k], (str, int)) and str(obj[k]).strip():
                        return str(obj[k]).strip()
                for v in obj.values():
                    found = seek_labels(v, keys, depth+1)
                    if found: return found
            elif isinstance(obj, list):
                for v in obj[:20]:
                    found = seek_labels(v, keys, depth+1)
                    if found: return found
            return None
        info = result.get("outAluno", {}) if isinstance(result, dict) else {}
        if isinstance(info, dict):
            series_value = info.get("outSerie")
            year_value = info.get("outAnoLetivo")
            if isinstance(series_value, int) and 1 <= series_value <= 9:
                user["school_series"] = series_value
            if isinstance(year_value, int) and 2000 <= year_value <= 2100:
                user["school_year"] = year_value
            # A resposta real de ObterInformacaoAluno traz outcdEscola, não o nome.
            school_id = info.get("outcdEscola")
            if isinstance(school_id, int) and school_id > 0:
                user["school_code"] = str(school_id)
        code = seek_labels(result, possible)
        if code and code.isascii() and code.isdecimal(): user["school_class_code"] = code
        name = seek_labels(result, school_keys)
        if name: user["school_name"] = name[:120]
        turma = seek_labels(result, class_keys)
        if turma: user["school_class_name"] = turma[:120]
    # Formato de faltas confirmado: {isSucess:true,data:[{totalFaltasBimestre,...}]}.
    # Retorne somente métricas comprovadas, sem identificadores do aluno.
    if recurso == "faltas":
        rows = result.get("data") if isinstance(result, dict) else None
        if not isinstance(rows, list) or not rows or result.get("isSucess") is not True:
            return {"ok": False, "status": "formato", "message": "A frequência não retornou o formato esperado."}
        item = rows[0]
        if not isinstance(item, dict):
            return {"ok": False, "status": "formato", "message": "Resposta de frequência inválida."}
        keys = ("totalFaltasBimestre", "totalAulasRealizadas", "porcentagemFaltas", "porcentagemFrequencia")
        if not all(isinstance(item.get(k), (int, float)) and not isinstance(item.get(k), bool) for k in keys):
            return {"ok": False, "status": "formato", "message": "Campos de frequência ausentes."}
        return {"ok": True, "status": "recebido", "recurso": "faltas", "dados": {k: item[k] for k in keys}}
    # O boletim retorna avaliações individuais (não médias bimestrais).
    # Preservar todos os registros, inclusive quando há mais de 60,
    # mas devolver ao navegador somente os campos necessários para exibição.
    if recurso == "boletim":
        records = result.get("data") if isinstance(result, dict) else None
        if not isinstance(records, list):
            return {"ok": False, "status": "formato", "message": "O serviço retornou um formato de boletim desconhecido."}
        safe_records = []
        for item in records[:3000]:
            if not isinstance(item, dict):
                continue
            safe_records.append({k: item.get(k) for k in (
                "disciplinaId", "bimestre", "descricaoAvaliacao", "peso",
                "notaAtribuida", "dataAvaliacao", "dataExclusao")})
        return {"ok": True, "status": "recebido", "recurso": "boletim", "dados": {"data": safe_records}}
    # O retorno é exibido SOMENTE ao navegador da sessão autenticada, após limpeza de PII.
    return {"ok": True, "status": "recebido", "recurso": recurso, "dados": _hyper_sanitize(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)
