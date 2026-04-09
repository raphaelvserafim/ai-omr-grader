from ultralytics import YOLO
import cv2

# 1. Carrega o modelo de bolinhas
model = YOLO('runs/detect/train2/weights/best.pt')

def processar_gabarito_inteligente(img_path):
    img = cv2.imread(img_path)
    
    # --- DETECTAR ORIENTAÇÃO PELO QR CODE ---
    qr_detector = cv2.QRCodeDetector()
    # Achamos o QR Code
    ok, points = qr_detector.detect(img)
    
    # Valor padrão: assume que o papel está de pé
    esta_invertido = False
    
    if ok and points is not None:
        # Pega o centro Y do QR Code
        qr_y = np.mean(points[0][:, 1])
        altura_img = img.shape[0]
        
        # Se o QR Code estiver na metade de baixo da imagem, está invertido
        if qr_y > (altura_img / 2):
            esta_invertido = True
            print("🔄 Papel detectado de cabeça para baixo. Corrigindo...")
        else:
            print("⬆️ Papel detectado na posição correta.")
    else:
        print("⚠️ QR Code não encontrado. Usando orientação padrão (em pé).")

    # 2. Rodar a detecção do YOLO
    results = model.predict(source=img, conf=0.3, max_det=40)
    
    deteccoes = []
    for r in results:
        for box in r.boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls = int(box.cls[0])
            deteccoes.append({'x': x, 'y': y, 'classe': cls})

    if len(deteccoes) != 40:
        return f"Erro: Detectadas {len(deteccoes)} bolinhas. Tire a foto novamente."

    # 3. Ordenar por Y (Cima para Baixo)
    # Se estiver invertido, a Q10 estará no topo, então invertemos a ordem
    deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d['y'])
    if esta_invertido:
        deteccoes_ordenadas.reverse()

    # 4. Definir Letras
    # Se estiver de pé: Esquerda -> Direita é D, C, B, A
    # Se estiver invertido: Esquerda -> Direita é A, B, C, D
    if not esta_invertido:
        letras = ['D', 'C', 'B', 'A']
    else:
        letras = ['A', 'B', 'C', 'D']

    gabarito_objeto = {}

    for i in range(0, len(deteccoes_ordenadas), 4):
        questao_num = (i // 4) + 1
        grupo_questao = deteccoes_ordenadas[i:i+4]
        
        # Ordena as 4 por X (Esquerda para Direita)
        grupo_ordenado_x = sorted(grupo_questao, key=lambda d: d['x'])
        
        marcadas = [idx for idx, b in enumerate(grupo_ordenado_x) if b['classe'] == 0]
        
        if len(marcadas) == 1:
            gabarito_objeto[questao_num] = letras[marcadas[0]]
        else:
            gabarito_objeto[questao_num] = None

    return gabarito_objeto

# Testar
import numpy as np
print(processar_gabarito_inteligente('duplicadovirada.png'))
print(processar_gabarito_inteligente('duplicado.png'))