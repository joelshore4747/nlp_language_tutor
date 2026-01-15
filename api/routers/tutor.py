from fastapi import APIRouter
from api.schemas import TutorEvaluateRequest, TutorEvaluateResponse
from api.services.tutor_service import tutor_evaluate as evaluate_service

router = APIRouter(prefix="/tutor", tags=["tutor"])

@router.post("/evaluate", response_model=TutorEvaluateResponse)
def tutor_evaluate(req: TutorEvaluateRequest) -> TutorEvaluateResponse:
    return evaluate_service(req)
