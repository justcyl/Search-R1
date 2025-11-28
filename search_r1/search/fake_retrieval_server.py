import argparse
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class SearchRequest(BaseModel):
    queries: List[str]


parser = argparse.ArgumentParser(description="Launch a fake retrieval server for debugging.")
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Server host.",
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Server port.",
)
args = parser.parse_args()

app = FastAPI(title="Fake Retrieval Server")


def _mock_result(query: str) -> dict:
    # 生成稳定且可读的假检索结果，便于调试训练/推理流程。
    return {
        "document": {
            "contents": f"\"Fake title for: {query}\"\nThis is a fake snippet for query [{query}].",
        }
    }


@app.post("/retrieve")
def retrieve(req: SearchRequest):
    # 对每个查询返回一条假结果，保持与真实接口相同的返回格式。
    return {"result": [[_mock_result(q)] for q in req.queries]}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
