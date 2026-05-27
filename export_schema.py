from models import RequestPayload, ResponsePayload
import json

json.dump({
    "request": RequestPayload.model_json_schema(),
    "response": ResponsePayload.model_json_schema()
}, open("schema.json", "w"), indent=2)