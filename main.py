from fastapi import FastAPI
from core.schemas import OMRRequest, OMRResponse
from core.omr_service import OMRService

app = FastAPI(title="AI OMR Grader API")

# Instancia o serviço globalmente (Singleton)
omr_service = OMRService(model_path='models/preta.azul.best.pt')

@app.get("/")
def health_check():
    return {"status": "online e funcionando!", "modelo": "preta.azul.best.pt", "datetime": datetime.now().isoformat()}

@app.post("/v1/decode-omr", response_model=OMRResponse)
async def decode_omr(request: OMRRequest):
    resultado = omr_service.processar_prova(request.image_base64)
    return resultado

if __name__ == "__main__":
    import uvicorn
    # Roda na porta 8000 
    uvicorn.run(app, host="0.0.0.0", port=8000)