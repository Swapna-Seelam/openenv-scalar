import uvicorn
from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app
from fastapi.responses import HTMLResponse
from env.email_env import EmailTriageEnv, Action, Observation

# Initialize the environment
env = EmailTriageEnv()

# Create the FastAPI app with the environment factory and models
app = create_fastapi_app(
    lambda: EmailTriageEnv(),
    action_cls=Action,
    observation_cls=Observation
)

@app.get("/grader")
async def get_grader_score():
    """Returns the current score clamped strictly between 0.01 and 0.99."""
    # Always return current progress score, no 'done' check as requested
    final_score = env.state.get("total_reward", 0.01)
    # Apply hard clamp
    final_score = max(0.01, min(0.99, float(final_score)))
    return {"score": final_score}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>OpenEnv Email Triage</title>
            <style>
                body { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                .card { background: white; padding: 2rem; border-radius: 1rem; shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border: 1px solid #e2e8f0; text-align: center; max-width: 400px; }
                h1 { color: #2563eb; margin-bottom: 0.5rem; }
                .code-box { background: #f1f5f9; padding: 1rem; border-radius: 0.5rem; text-align: left; font-family: monospace; font-size: 0.875rem; margin-top: 1rem; }
                .status { display: inline-block; background: #dcfce7; color: #166534; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; }
                a { color: #2563eb; text-decoration: none; font-weight: 500; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="card">
                <div class="status">● System Online</div>
                <h1>OpenEnv Scalar</h1>
                <p>Email Triage Environment Server</p>
                <div class="code-box">
                    POST /reset<br>
                    POST /step<br>
                    GET /state<br>
                    GET /docs
                </div>
                <p style="margin-top: 1.5rem;"><a href="/docs">View API Documentation &rarr;</a></p>
            </div>
        </body>
    </html>
    """

def main():
    # Use standard Hugging Face Spaces port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
