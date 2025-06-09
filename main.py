import cv2
import math
from ultralytics import YOLO
import time # Mantido caso queira adicionar delays ou medições de FPS

# --- CONFIGURAÇÕES ---
WEBCAM_ID = 0  # ID da sua webcam (normalmente 0 para física, 2 ou mais para OBS)
MODEL_PATH = "datsetpropriov2.pt"  # Caminho para o seu modelo treinado
MIN_CONFIDENCE = 0.60  # Confiança mínima para considerar uma detecção válida

# Nomes das classes - DEVEM CORRESPONDER EXATAMENTE À ORDEM DO SEU MODELO TREINADO
CLASS_NAMES = ['capacete', 'colete', 'luvas', 'mascara', 'oculos']

# --- CONFIGURAÇÕES DA JANELA DE EXIBIÇÃO ---
WINDOW_NAME = 'Validacao de TODOS os EPIs (CPU)'
# Defina a largura e altura desejadas para a janela de exibição
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# --- INICIALIZAÇÃO ---

# Webcam
print(f"Tentando abrir a webcam ID: {WEBCAM_ID}...")
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir a webcam ID {WEBCAM_ID}. Verifique se está conectada, não em uso, e se a câmera virtual do OBS está iniciada.")
    exit()
# Define a resolução da captura da webcam (o modelo YOLO pode redimensionar internamente)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Webcam aberta com sucesso.")

# Modelo YOLO
print(f"Carregando modelo YOLO de: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    #model.to('cpu') # Força o modelo a usar a CPU
    print(f"Modelo YOLO '{MODEL_PATH}' carregado e configurado para rodar na CPU. Dispositivo: {model.device}")
except Exception as e:
    print(f"ERRO ao carregar o modelo YOLO '{MODEL_PATH}': {e}")
    cap.release()
    exit()

# --- CRIAR E CONFIGURAR A JANELA DE EXIBIÇÃO ---
# Crie uma janela nomeada que possa ser redimensionada
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# Redimensione a janela para as dimensões desejadas
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
print(f"Janela de exibicao '{WINDOW_NAME}' configurada para {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}.")

# --- LOOP PRINCIPAL DE PROCESSAMENTO ---
print("Iniciando processamento da webcam (usando CPU)...")
while True:
    success, img = cap.read()
    if not success:
        print("ERRO: Não foi possível ler o frame da webcam. Encerrando.")
        break

    results = model(img, stream=True, verbose=False)
    detected_ppe_names_in_frame = set()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = math.ceil(box.conf[0] * 100) / 100
            if confidence < MIN_CONFIDENCE:
                continue

            cls_index = int(box.cls[0])
            if cls_index < 0 or cls_index >= len(CLASS_NAMES):
                print(f"Aviso: Índice de classe {cls_index} inválido para CLASS_NAMES. Pulando esta detecção.")
                continue
            current_class_name = CLASS_NAMES[cls_index]
            detected_ppe_names_in_frame.add(current_class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{current_class_name}: {confidence:.2f}"
            text_y_pos = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(img, label, (x1, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if detected_ppe_names_in_frame:
        summary_text = f"Detectado: {', '.join(sorted(list(detected_ppe_names_in_frame)))}"
    else:
        summary_text = "Nenhum EPI detectado"
    
    (text_width, text_height), baseline = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (5, 5), (10 + text_width, 10 + text_height + baseline), (0,0,0), cv2.FILLED)
    cv2.putText(img, summary_text, (10, 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Exibir a imagem processada na janela nomeada
    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Encerrando...")
        break

# --- Finalização ---
print("Liberando recursos...")
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
