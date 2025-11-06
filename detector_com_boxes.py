import os
import threading
import time
from collections import deque
from datetime import datetime
from queue import Queue
import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO


class DetectorComBoxes:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)

        # Configurações de performance da câmera
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("🔄 Carregando YOLO...")
        self.yolo = YOLO('yolov8n.pt')
        # ✅ YOLO na CPU (mais fluido para detecção em tempo real)
        self.yolo.to('cpu')
        print("✅ YOLO carregado na CPU!\n")

        print("🔄 Carregando VideoMAE...")
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)
        print(f"✅ VideoMAE em '{self.device}'.\n")

        self.frame_buffer = deque(maxlen=16)

        # Sistema de threading
        self.fila_analise = Queue(maxsize=1)
        self.analisando = False
        self.thread_analise = None
        self._iniciar_worker_analise()

        # Gravação
        self.gravando = False
        self.inicio_gravacao = None
        self.video_writer = None
        self.duracao_gravacao = 10
        self.pasta_videos = "videos_anomalias"
        if not os.path.exists(self.pasta_videos):
            os.makedirs(self.pasta_videos)

        # Flask
        self.ultimo_frame = None
        self.ultima_deteccao = None
        self.historico_videos = []
        self.frame_anterior = None
        self.rodando = False

        # YOLO otimizado
        self.contador_frames = 0
        self.intervalo_yolo = 10  # Pode ser mais frequente agora na CPU
        self.num_pessoas = 0
        self.boxes_yolo = []

        # Cache de movimento
        self.ultimo_movimento_time = 0
        self.movimento_cooldown = 2

    def _iniciar_worker_analise(self):
        """Inicia worker thread para análise em background"""

        def worker():
            while True:
                try:
                    video_clip, frame_atual = self.fila_analise.get()
                    if video_clip is None:
                        break
                    self._analisar_clipe(video_clip, frame_atual)
                    self.fila_analise.task_done()
                except Exception as e:
                    print(f"❌ Erro no worker: {e}")
                    self.analisando = False

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _analisar_clipe(self, video_clip, frame_atual):
        """Função interna para análise (chamada pelo worker)"""
        try:
            print("📸 Analisando movimento...")
            label, confianca = self.classificar_video(video_clip)
            print(f"🔎 IA: '{label}' ({confianca:.1%})")

            evento = None
            if any(palavra in label for palavra in ["fight", "punch", "kick", "hit"]):
                evento = "violencia_detectada"
            elif any(palavra in label for palavra in ["running", "jumping", "falling", "climbing"]):
                evento = "comportamento_suspeito"
            elif any(palavra in label for palavra in ["robbery", "burglary", "stealing"]):
                evento = "atividade_ilicita"

            self.ultima_deteccao = {
                'acao': label,
                'evento': evento,
                'confianca': f"{confianca:.1%}",
                'timestamp': datetime.now()
            }

            if evento:
                print(f"🚨 Evento: {evento}")
                self.iniciar_gravacao(frame_atual, evento)

        except Exception as e:
            print(f"❌ Erro na análise: {e}")
        finally:
            self.analisando = False

    def detectar_movimento(self, frame1, frame2, limiar_area=1500):
        """Detecta movimento otimizado"""
        if frame1 is None or frame2 is None:
            return False

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            if cv2.contourArea(c) > limiar_area:
                return True
        return False

    def processar_yolo_rapido(self, frame):
        """YOLO na CPU - mais fluido para tempo real"""
        # ✅ YOLO na CPU não precisa de conversão de tensor
        results = self.yolo(frame, verbose=False, conf=0.5, imgsz=320)

        pessoas = 0
        boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    pessoas += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

        return pessoas, boxes

    def desenhar_deteccoes(self, frame):
        """Desenho otimizado"""
        for i, (x1, y1, x2, y2) in enumerate(self.boxes_yolo, 1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'P{i}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame

    def classificar_video(self, video_clip):
        """Classificação na GPU (VideoMAE)"""
        # Reduz qualidade dos frames para análise mais rápida
        video_clip_otimizado = []
        for frame in video_clip:
            frame_pequeno = cv2.resize(frame, (224, 224))
            video_clip_otimizado.append(frame_pequeno)

        inputs = self.processor(video_clip_otimizado, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = logits.argmax(-1).item()
            confianca = probs[0][pred_idx].item()
            label = self.model.config.id2label[pred_idx]

        return label.lower(), confianca

    def iniciar_gravacao(self, frame, evento_detectado):
        """Inicia gravação"""
        if self.gravando:
            return

        altura, largura = frame.shape[:2]
        timestamp = int(time.time())
        nome = f"{evento_detectado}_{timestamp}.mp4"
        caminho = os.path.join(self.pasta_videos, nome)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(caminho, fourcc, 15.0, (largura, altura))

        self.gravando = True
        self.inicio_gravacao = time.time()

        self.historico_videos.append({
            'nome': nome,
            'caminho': caminho,
            'evento': evento_detectado,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        })

        print(f"🎥 Gravando: {nome}")

    def processar_frame(self):
        """Processamento otimizado - YOLO na CPU"""
        if not self.rodando:
            return None

        # Limpa buffer da câmera
        for _ in range(2):
            ret, frame_atual = self.cap.read()
            if not ret:
                return None

        # YOLO na CPU (pode ser mais frequente)
        self.contador_frames += 1
        if self.contador_frames % self.intervalo_yolo == 0:
            self.num_pessoas, self.boxes_yolo = self.processar_yolo_rapido(frame_atual)

        # Desenha boxes
        if self.boxes_yolo:
            self.desenhar_deteccoes(frame_atual)

        # Informações visuais
        cv2.putText(frame_atual, f"P: {self.num_pessoas}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if self.analisando:
            cv2.putText(frame_atual, "ANALISANDO", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.gravando:
            tempo_restante = int(self.duracao_gravacao - (time.time() - self.inicio_gravacao))
            cv2.putText(frame_atual, f"REC {tempo_restante}s", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Detecção de movimento
        if self.frame_anterior is not None:
            tempo_atual = time.time()
            if tempo_atual - self.ultimo_movimento_time > self.movimento_cooldown:
                movimento = self.detectar_movimento(self.frame_anterior, frame_atual)

                if movimento and not self.gravando and len(self.frame_buffer) == 16 and not self.analisando:
                    self.analisando = True
                    self.ultimo_movimento_time = tempo_atual

                    video_clip_copy = list(self.frame_buffer)
                    frame_copy = frame_atual.copy()

                    try:
                        self.fila_analise.put_nowait((video_clip_copy, frame_copy))
                    except:
                        self.analisando = False

        self.frame_buffer.append(frame_atual)

        # Gravação
        if self.gravando:
            self.video_writer.write(frame_atual)
            if time.time() - self.inicio_gravacao >= self.duracao_gravacao:
                self.video_writer.release()
                self.gravando = False
                print("💾 Gravação finalizada")

        self.frame_anterior = frame_atual.copy()
        self.ultimo_frame = frame_atual

        return frame_atual

    def iniciar(self):
        """Inicia o sistema"""
        print("🔵 Iniciando sistema...")

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)

        self.rodando = True
        time.sleep(0.5)

        ret, frame = self.cap.read()
        if frame is not None:
            print("✅ Sistema iniciado!")
            for _ in range(16):
                self.frame_buffer.append(frame)
            self.frame_anterior = frame
        else:
            print("❌ Erro ao capturar frame")

    def parar(self):
        """Para o sistema"""
        print("🔴 Parando sistema...")
        self.rodando = False

        if self.gravando and self.video_writer:
            self.video_writer.release()
            self.gravando = False

        if self.cap.isOpened():
            self.cap.release()

    def get_historico_videos(self):
        """Retorna lista de vídeos"""
        return sorted(self.historico_videos, key=lambda x: x['timestamp'], reverse=True)