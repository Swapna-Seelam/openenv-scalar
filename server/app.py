import uvicorn
from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app
from env.email_env import EmailTriageEnv, Action, Observation

# Initialize the environment
env = EmailTriageEnv()

# Create the FastAPI app with the environment factory and models
app = create_fastapi_app(
    lambda: EmailTriageEnv(),
    action_cls=Action,
    observation_cls=Observation
)

def main():
    # Use standard Hugging Face Spaces port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
