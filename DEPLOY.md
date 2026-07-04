# Deploying the Clinical Insight Agent

The app is a containerized Streamlit service (`Dockerfile`). It needs one runtime secret,
`OPENAI_API_KEY`, and it serves on `$PORT` (default `8080`). The committed slim demo warehouse
(`data/healthcare_demo.duckdb`) ships in the image, so there is nothing else to provision.

> Why a container (vs. Streamlit Community Cloud): the image is **built once and cached**, so
> redeploys are layer-diffs that go live in seconds, there is no revert-to-private quirk, and you
> get a custom domain. Streamlit is a stateful server, so it does **not** run on Vercel — a container
> host is the right target.

---

## Option A — Google Cloud Run  (recommended: custom domain, scales to zero, near-free)

```bash
# one-time
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# store the key in Secret Manager (don't pass it inline)
printf '%s' "sk-...your-key..." | gcloud secrets create openai-api-key --data-file=-

# build (from the Dockerfile) + deploy
gcloud run deploy clinical-insight-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 --memory 2Gi --cpu 2 --timeout 300 \
  --min-instances 0 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

Cloud Run prints a `*.run.app` URL. Redeploy anytime with the same `gcloud run deploy` line.
Custom domain: `gcloud run domain-mappings create --service clinical-insight-agent --domain agent.yourdomain.com`.
`--min-instances 0` scales to zero (pay ~nothing idle); use `1` to avoid cold starts.

## Option B — Hugging Face Spaces (Docker)  (fastest: free, no billing, credible ML home)

1. Create a Space at <https://huggingface.co/new-space> → **SDK: Docker**.
2. Prepend this front-matter to the Space's `README.md`:
   ```
   ---
   title: Clinical Insight Agent
   emoji: 🩺
   sdk: docker
   app_port: 8080
   ---
   ```
3. Push the code to the Space and add the secret:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USER/clinical-insight-agent
   git push space main
   ```
   Then **Settings → Variables and secrets → New secret**: `OPENAI_API_KEY = sk-...`.

Free tier: 2 vCPU / 16 GB, sleeps after inactivity and wakes on visit. No card required.

## Option C — Render  (simple Git-connected Docker deploy)

New **Web Service** → connect the repo → Runtime **Docker** → add env var `OPENAI_API_KEY` →
create. Render injects `$PORT` automatically. Free tier spins down when idle.

---

## Test the image locally (any host)

```bash
docker build -t clinical-agent .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=sk-... clinical-agent
# open http://localhost:8080
```

## Notes
- **Never** bake the key into the image or commit `agent/.env` (it is git- and docker-ignored).
- The full 200 MB warehouse is excluded via `.dockerignore`; the app queries the slim demo DB.
- To refresh the demo data after dbt model changes, rebuild it and re-commit (see the project runbook).
