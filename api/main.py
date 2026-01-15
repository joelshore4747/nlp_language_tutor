from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.semantic import router as semantic_router
from api.routers.tutor import router as tutor_router

def create_app() -> FastAPI:
    app = FastAPI(title="NLP Adaptive Tutor API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(semantic_router)
    app.include_router(tutor_router)

    return app

app = create_app()
