# Hyper Futuro X — Backend para Render

Arquivos extraídos do ZIP original enviado pelo proprietário. Não inclui `.env`, senhas pessoais ou banco de dados local.

## Subir no GitHub
Abra um repositório seu, use Add file > Upload files e carregue o **conteúdo extraído** deste ZIP (não coloque uma pasta extra acima de `app.py`).

## Configurar Web Service no Render
Runtime: Python 3
Build Command: `pip install -r requirements.txt`
Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Observações importantes:
- O projeto original inclui PyTorch, torchvision e modelo de visão; o build e a RAM podem exceder o limite de planos gratuitos.
- Configure segredos apenas em Environment no Render; não publique .env ou credenciais em GitHub.
- O login depende de serviços de terceiros, disponibilidade de APIs e cookies/sessões. Publicar não garante que funcionará.
- Frontend hospedado em outro domínio requer configurar acesso seguro entre origens e sessão/cookies; não exponha credenciais de usuários.
- Banco SQLite local no Render pode ser perdido quando a instância reiniciar/reimplantar, sem disco persistente.
- Antes de expor a aplicação a terceiros, revise autenticação, segurança e privacidade.
