import io
import json
import zipfile
from fastapi import APIRouter,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel,Field,model_validator
from typing import Literal
from .main import Document,Experiment,Rule
from . import lab
router=APIRouter(prefix="/api/lab")

class Query(BaseModel):
    id:str=Field(min_length=1,max_length=100)
    text:str=Field(min_length=1,max_length=4000)
    relevant:list[str]=Field(min_length=1)
    rules:list[Rule]=Field(default_factory=list,max_length=6)

class Dataset(BaseModel):
    documents:list[Document]=Field(min_length=2,max_length=200)
    queries:list[Query]=Field(min_length=1,max_length=20)
    models:list[str]=Field(min_length=2,max_length=2)
    k:int=Field(default=1,ge=1,le=200)
    budget_per_query:int=Field(default=63,ge=1,le=63)
    @model_validator(mode="after")
    def validate_data(self):
        ids=[q.id for q in self.queries]
        if len(set(ids))!=len(ids):raise ValueError("Query IDs must be unique")
        for q in self.queries:
            Experiment(query=q.text,variant=q.text,kind="meaning_preserving",documents=self.documents,
                       relevant_before=q.relevant,relevant_after=q.relevant,models=self.models,k=self.k)
            if any(r.source==r.replacement for r in q.rules):raise ValueError("Rules must change text")
            lab.edits(q.text,[r.model_dump() for r in q.rules])
        return self

class Review(BaseModel):
    status:Literal["accepted","rejected","uncertain"]
    note:str=Field(min_length=1,max_length=2000)

def execute(fn,*args):
    try:return fn(*args)
    except ValueError as e:raise HTTPException(422,str(e))
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Lab operation failed")
        raise HTTPException(503,"Operation failed. Check server logs and model dependencies.") from e

@router.post("/validate")
def validate(body:Dataset):return {"dataset":body.model_dump(),"document_count":len(body.documents),"query_count":len(body.queries)}
@router.post("/search")
def search(body:Dataset):return execute(lab.search,body.model_dump())
@router.get("/runs/{rid}")
def get(rid:str):return execute(lab.get,rid)
@router.post("/runs/{rid}/candidates/{cid}/review")
def review(rid:str,cid:str,body:Review):return execute(lab.review,rid,cid,body.status,body.note)
@router.post("/runs/{rid}/candidates/{cid}/reduce")
def reduce(rid:str,cid:str):return execute(lab.reduce,rid,cid)
@router.get("/runs/{rid}/candidates/{cid}/export")
def export(rid:str,cid:str):
    bundle=execute(lab.export,rid,cid);buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("case.json",json.dumps(bundle["case"],ensure_ascii=False,indent=2))
        z.writestr("test_regression.py",bundle["pytest"])
        z.writestr("README.txt",bundle["instructions"])
    return Response(buffer.getvalue(),media_type="application/zip")
