from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from .engine import run_experiment, DEMO_MODELS, REAL_MODELS

class Document(BaseModel):
    id: str = Field(min_length=1,max_length=100)
    text: str = Field(min_length=1,max_length=10000)

class Experiment(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    variant: str = Field(min_length=1,max_length=4000)
    kind: Literal["meaning_preserving","condition_change"]
    factor: str = Field(default="custom",max_length=100)
    documents: list[Document] = Field(min_length=2,max_length=200)
    relevant_before: list[str] = Field(min_length=1)
    relevant_after: list[str] = Field(min_length=1)
    models: list[str] = Field(min_length=2,max_length=2)
    k: int = Field(ge=1,le=200)

    @model_validator(mode="after")
    def valid(self):
        ids=[d.id for d in self.documents]
        if len(set(ids)) != len(ids): raise ValueError("Document IDs must be unique")
        if self.k > len(ids): raise ValueError("k exceeds corpus size")
        for labels in [self.relevant_before,self.relevant_after]:
            if len(set(labels))!=len(labels) or not set(labels)<=set(ids):
                raise ValueError("Labels must be unique document IDs from the corpus")
        if len(set(self.models))!=2 or not set(self.models)<=DEMO_MODELS|REAL_MODELS:
            raise ValueError("Choose two different supported models")
        if self.kind == "meaning_preserving" and set(self.relevant_before)!=set(self.relevant_after):
            raise ValueError("Meaning-preserving experiments require unchanged labels")
        return self

app=FastAPI(title="Embedding Microscope")
static=Path(__file__).parent/"static"
app.mount("/static",StaticFiles(directory=static),name="static")
@app.get("/")
def index(): return FileResponse(static/"lab.html")
@app.get("/experiment")
def single_experiment(): return FileResponse(static/"index.html")
@app.get("/api/models")
def models(): return {"demo":sorted(DEMO_MODELS),"local":sorted(REAL_MODELS)}
@app.post("/api/experiment")
def experiment(body: Experiment):
    try: return run_experiment(body.model_dump())
    except ImportError: raise HTTPException(503,"Install model dependencies: pip install -e '.[models]'")
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Experiment failed")
        raise HTTPException(503,"Model execution failed. Check server logs and model download connectivity.") from e

class Rule(BaseModel):
    source: str = Field(min_length=1,max_length=100)
    replacement: str = Field(min_length=1,max_length=100)

class Hunt(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    documents: list[Document] = Field(min_length=2,max_length=200)
    relevant: list[str] = Field(min_length=1)
    models: list[str] = Field(min_length=2,max_length=2)
    rules: list[Rule] = Field(min_length=1,max_length=6)
    k: int = Field(ge=1,le=200)
    budget: int = Field(default=63,ge=1,le=63)
    approved: bool

    @model_validator(mode="after")
    def valid(self):
        if not self.approved: raise ValueError("Review relevance preservation before searching")
        if any(r.source == r.replacement for r in self.rules): raise ValueError("Rules must change text")
        Experiment(query=self.query,variant=self.query,kind="meaning_preserving",
                   documents=self.documents,relevant_before=self.relevant,relevant_after=self.relevant,
                   models=self.models,k=self.k)
        return self

@app.post("/api/hunt")
def hunt(body: Hunt):
    from .hunt import search
    try: return search(body.model_dump())
    except ValueError as e: raise HTTPException(422,str(e))
    except ImportError: raise HTTPException(503,"Install model dependencies: pip install -e '.[models]'")
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Hunt failed")
        raise HTTPException(503,"Model execution failed. Check server logs.") from e

@app.post("/api/hunt/export")
def export_hunt(body: Hunt):
    from .hunt import regression_bundle
    try:
        import io, json, zipfile
        from fastapi.responses import Response
        bundle=regression_bundle(hunt(body))
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("case.json",json.dumps(bundle["case"],ensure_ascii=False,indent=2))
            archive.writestr("test_regression.py",bundle["pytest"])
            archive.writestr("README.txt",bundle["instructions"])
        return Response(buffer.getvalue(),media_type="application/zip",headers={"Content-Disposition":"attachment; filename=retrieval-regression.zip"})
    except ValueError as e: raise HTTPException(422,str(e))

from .lab_api import router as lab_router
app.include_router(lab_router)
