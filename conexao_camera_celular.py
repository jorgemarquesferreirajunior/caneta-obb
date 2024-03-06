import cv2
import os 

HOME = os.getcwd()

def escalar_img(escala, f):
    h,w = f.shape[:2]
    prop = w / h
    new_h = int(h * escala)
    new_w = int(prop * new_h)
    f = cv2.resize(f, (new_w, new_h))
    
    return f    
    
    
    
video = cv2.VideoCapture("http://192.168.25.140:8080/video")

if not video.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    sucesso, frame = video.read()
    
    if not sucesso:
        print("Erro ao ler o frame.")
    
    frame = escalar_img(0.5, frame)
    # print(f'w: {frame.shape[1]} h: {frame.shape[0]}')
    h,w = frame.shape[:2]
    
    x = 200
    frame_alvo = frame.copy()
    cv2.rectangle(frame, (w//2-x, h//2-x), (w//2+x, h//2+x), (0, 255, 0), 3)
    cv2.circle(frame, (w//2, h//2), 10, (0,0,255), cv2.FILLED)
    cv2.imshow('Camera Xiaomi', frame)
    
    
    if cv2.waitKey(1) == ord('p'):
        regiao_interesse = frame_alvo[h//2-x:h//2+x, w//2-x:w//2+x]
        x = len(os.listdir(os.path.join(HOME, 'printscreens')))
        cv2.imwrite(os.path.join(HOME, 'printscreens', f'{str(x).zfill(2)}.jpg'), regiao_interesse)

    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()