from ultralytics import YOLO
from .utils import ImageUtils 

class OMRService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def processar_prova(self, b64_image: str):
        img = ImageUtils.base64_to_cv2(b64_image)
        if img is None:
            return {"status": "erro", "erro": "Imagem inválida"}

        esta_invertido = ImageUtils.check_orientation_qr(img)
        
        # Predição: classe 0 = marcada, classe 1 = não marcada (ajuste conforme seu modelo)
        results = self.model.predict(source=img, conf=0.3, max_det=40, verbose=False)
        
        deteccoes = []
        for r in results:
            for box in r.boxes:
                x, y, w, h = box.xywh[0].tolist()
                deteccoes.append({'x': x, 'y': y, 'classe': int(box.cls[0])})

        if len(deteccoes) != 40:
            return {"status": "erro", "erro": f"Detectadas {len(deteccoes)} bolinhas, esperado 40."}

        # 1. ORDENAÇÃO POR LINHAS (Eixo Y)
        # Se estiver invertido, a Questão 1 está no "fundo" da imagem (Y alto), 
        # então ordenamos de forma decrescente para que ela venha primeiro.
        deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d['y'], reverse=esta_invertido)

        # 2. DEFINIÇÃO DO MAPEAMENTO DE LETRAS
        # Se NÃO está invertido: da esquerda para direita é A, B, C, D
        # Se ESTÁ invertido: a imagem está de ponta-cabeça, então o que era 'D' 
        # na folha física agora aparece primeiro na esquerda da imagem digital.
        letras = ['A', 'B', 'C', 'D'] if not esta_invertido else ['D', 'C', 'B', 'A']

        gabarito = {}

        # 3. PROCESSAMENTO EM BLOCOS DE 4 (Cada questão)
        for i in range(0, 40, 4):
            questao_num = (i // 4) + 1
            
            # Pega as 4 bolinhas da linha atual e ordena por X (esquerda para direita)
            linha = sorted(deteccoes_ordenadas[i:i+4], key=lambda d: d['x'])
            
            # Identifica quais estão marcadas (supondo classe 0 como marcada)
            marcadas = [idx for idx, b in enumerate(linha) if b['classe'] == 0]
            
            # Regra: se houver exatamente uma marcada, registra. Caso contrário, None (rasura ou em branco)
            if len(marcadas) == 1:
                gabarito[questao_num] = letras[marcadas[0]]
            else:
                gabarito[questao_num] = None

        return {"status": "sucesso", "gabarito": gabarito}