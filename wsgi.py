# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chebifier[ensemble,web]",
#     "chemlog-extra",
#     "chebai-graph",
# ]
#
# [tool.uv.sources]
# chebifier = { path = ".", editable = true }
# chemlog-extra = { git = "https://github.com/ChEB-AI/chemlog-extra.git" }
# chebai-graph = { git = "https://github.com/ChEB-AI/python-chebai-graph.git" }
#
# [tool.uv.extra-build-dependencies]
# torch-scatter = ["torch"]
# ///


from chebifier import BaseEnsemble
from fastapi import FastAPI, APIRouter, Depends, Request
from typing import Annotated
from pydantic import BaseModel, Field
from curies import Reference

from fastapi.responses import RedirectResponse

router = APIRouter()



class PredictionRequest(BaseModel):
    """A request that includes a list of molecules, given by SMILES strings."""
    molecules: list[str] = Field(
        ...,
        description="A list of SMILES strings for molecules to predict",
        examples=[["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C(=O)O"]],
    )


class PredictionResponse(BaseModel):
    """A response that maps SMILES strings to ChEBI class references."""
    results: dict[str, list[Reference]] = Field(
        ...,
        description="A mapping from smiles"
    )


def get_ensemble(request: Request) -> BaseEnsemble:
    """Retrieve an ensemble from the app."""
    return request.app.state.ensemble


@router.post("/api/predict")
def predict(ensemble: Annotated[BaseEnsemble, Depends(get_ensemble)], prediction_request: PredictionRequest):
    """Predict ChEBI classes for given molecules by SMILES."""
    predictions = ensemble.predict_smiles_list(prediction_request.molecules)
    return PredictionResponse(
        results={
            smiles: _curify(p)
            for smiles, p in zip(prediction_request.molecules, predictions)
        }
    )


def _curify(chebi_ids: list[str]) -> list[Reference]:
    return [Reference(prefix="CHEBI", identifier=chebi_id) for chebi_id in chebi_ids]


def get_app() -> FastAPI:
    """Construct a Chebifier FastAPI app."""
    app = FastAPI(
        title="Chebifier App",
        description="A FastAPI-based web application wrapping Chebifier's ensemble predictor",
    )

    @app.get("/", include_in_schema=False)
    async def redirect_docs():
        return RedirectResponse("/docs")

    app.state.ensemble = BaseEnsemble()

    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(get_app())
