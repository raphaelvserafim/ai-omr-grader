from ultralytics import YOLO
from .utils import ImageUtils

class OMRService:
    def __init__(self, model_path: str):
        # Carrega o modelo apenas uma vez na inicialização
        self.model = YOLO(model_path)

    def processar_prova(self, b64_image: str):
        img = ImageUtils.base64_to_cv2(b64_image)
        if img is None:
            return {"status": "erro", "erro": "Imagem inválida"}

        esta_invertido = ImageUtils.check_orientation_qr(img)
        
        # Predição
        results = self.model.predict(source=img, conf=0.3, max_det=40, verbose=False)
        
        deteccoes = []
        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywh[0].tolist()
                deteccoes.append({'x': x, 'y': y, 'classe': int(box.cls[0])})

        if len(deteccoes) != 40:
            return {"status": "erro", "erro": f"Detectadas {len(deteccoes)} bolinhas."}

        # Ordenação e Lógica de Gabarito
        deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d['y'])
        if esta_invertido:
            deteccoes_ordenadas.reverse()

        letras = ['D', 'C', 'B', 'A'] if not esta_invertido else ['A', 'B', 'C', 'D']
        gabarito = {}

        for i in range(0, len(deteccoes_ordenadas), 4):
            questao_num = (i // 4) + 1
            grupo = sorted(deteccoes_ordenadas[i:i+4], key=lambda d: d['x'])
            marcadas = [idx for idx, b in enumerate(grupo) if b['classe'] == 0]
            
            gabarito[questao_num] = letras[marcadas[0]] if len(marcadas) == 1 else None

        return {"status": "sucesso", "gabarito": gabarito}