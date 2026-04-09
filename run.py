from ultralytics import YOLO
import cv2
import json

# 1. Carrega o modelo
model = YOLO('runs/detect/train2/weights/best.pt')

# 2. Roda a detecção (max_det=40 para pegar todas as bolinhas)
img_path = 'prateste.png'
results = model.predict(source=img_path, conf=0.3, max_det=40)

deteccoes = []
for r in results:
    # Salva imagem para conferência visual
    cv2.imwrite('conferencia_final.png', r.plot())
    for box in r.boxes:
        x, y, w, h = box.xywh[0].tolist()
        cls = int(box.cls[0]) # 0: marcado, 1: vazio
        deteccoes.append({'x': x, 'y': y, 'classe': cls})

# Verifica se o modelo encontrou as 40 bolinhas
if len(deteccoes) != 40:
    print(f"⚠️ Aviso: Foram detectadas {len(deteccoes)} bolinhas. O resultado pode falhar.")

# 3. Ordenar por Y (cima para baixo) para separar as linhas das questões
deteccoes_ordenadas = sorted(deteccoes, key=lambda d: d['y'])

# 4. Processar o Gabarito
letras = ['D', 'C', 'B', 'A'] # Ordem da esquerda para a direita na foto
gabarito_objeto = {}

# Loop de 4 em 4 bolinhas (10 questões)
for i in range(0, len(deteccoes_ordenadas), 4):
    questao_num = (i // 4) + 1
    # Pega o bloco de 4 bolinhas da questão atual
    grupo_questao = deteccoes_ordenadas[i:i+4]
    
    # Ordena as 4 bolinhas por X (esquerda para direita) para saber qual é D, C, B, A
    grupo_ordenado_x = sorted(grupo_questao, key=lambda d: d['x'])
    
    # Filtra apenas as que o modelo disse que estão MARCADAS (classe 0)
    marcadas = [idx for idx, b in enumerate(grupo_ordenado_x) if b['classe'] == 0]
    
    # LÓGICA DE VALIDAÇÃO:
    # Se houver EXATAMENTE uma marcada, pegamos a letra.
    # Se houver 0 ou mais de 1, definimos como None (null).
    if len(marcadas) == 1:
        gabarito_objeto[questao_num] = letras[marcadas[0]]
    else:
        gabarito_objeto[questao_num] = None

# 5. Resultado Final
print(gabarito_objeto)
 