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
import json
from config_loader import ConfigLoader


class DetectorComBoxes:
    def __init__(self, video_source=0, camera_id=0, camera_nome="Camera", config=None):
        # Carregar configurações
        self.config = config if config else ConfigLoader()
        self.camera_id = camera_id
        self.camera_nome = camera_nome

        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)

        # Configurações de performance da câmera (do config)
        self.fps_camera = self.config.get('sistema', 'fps_camera', default=30)
        largura = self.config.get('sistema', 'resolucao', 'largura', default=640)
        altura = self.config.get('sistema', 'resolucao', 'altura', default=480)

        self.cap.set(cv2.CAP_PROP_FPS, self.fps_camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Carregar modelos IA
        usar_gpu = self.config.get('sistema', 'usar_gpu', default=True)
        modelo_yolo = self.config.get('deteccao', 'modelo_yolo', default='yolov8n.pt')
        modelo_videomae = self.config.get('deteccao', 'modelo_videomae', default='MCG-NJU/videomae-base-finetuned-kinetics')

        # Determinar device ANTES de carregar os modelos
        self.device = torch.device("cuda" if (usar_gpu and torch.cuda.is_available()) else "cpu")

        print(f"🔄 [{self.camera_nome}] Carregando YOLO...")
        # YOLO carrega direto no device correto
        self.yolo = YOLO(modelo_yolo)
        # Configurar device para YOLO nas predições
        self.yolo_device = 0 if (usar_gpu and torch.cuda.is_available()) else 'cpu'
        print(f"✅ [{self.camera_nome}] YOLO configurado para {self.device}!\n")

        print(f"🔄 [{self.camera_nome}] Carregando VideoMAE...")
        self.processor = VideoMAEImageProcessor.from_pretrained(modelo_videomae)
        self.model = VideoMAEForVideoClassification.from_pretrained(modelo_videomae).to(self.device)
        print(f"✅ [{self.camera_nome}] VideoMAE em '{self.device}'.\n")

        buffer_size = self.config.get('sistema', 'buffer_frames', default=16)
        self.frame_buffer = deque(maxlen=buffer_size)

        # Sistema de threading
        self.fila_analise = Queue(maxsize=1)
        self.analisando = False
        self.thread_analise = None
        self._iniciar_worker_analise()

        # SISTEMA DUAL DE GRAVAÇÃO
        # 1. Gravação contínua (sessão completa)
        self.gravacao_continua = False
        self.gravador_continuo = None
        self.inicio_sessao = None
        self.nome_video_continuo = None

        # 2. Clips de eventos (configurável)
        self.gravando = False
        self.inicio_gravacao = None
        self.video_writer = None
        self.duracao_gravacao = self.config.get('gravacao', 'duracao_clip_anomalia', default=10)

        # 3. Pastas e índice (configurável)
        self.pasta_videos = self.config.get('gravacao', 'pasta_videos_anomalias', default='videos_anomalias')
        self.pasta_sessoes = self.config.get('gravacao', 'pasta_videos_sessoes', default='videos_sessoes')
        self.anomalias_detectadas = []
        self.arquivo_indice = None

        # Criar pastas com ID da câmera
        self.pasta_videos = os.path.join(self.pasta_videos, f"camera_{self.camera_id}")
        self.pasta_sessoes = os.path.join(self.pasta_sessoes, f"camera_{self.camera_id}")

        for pasta in [self.pasta_videos, self.pasta_sessoes]:
            if not os.path.exists(pasta):
                os.makedirs(pasta)

        # Flask
        self.ultimo_frame = None
        self.ultima_deteccao = None
        self.historico_videos = []
        self.frame_anterior = None
        self.rodando = False

        # YOLO otimizado
        self.contador_frames = 0
        self.intervalo_yolo = 10
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
            label_ingles, confianca = self.classificar_video(video_clip)

            # --- MUDANÇA 1: Dicionário de Tradução/Personalização ---
            traducoes = {
                "beatboxing": "Movimento Brusco",   
                "unboxing" : "agressão",   
                "punching bag": "Vandalismo",     
                "punching person (boxing)": "Soco Detectado",
                "headbutting": "Cabeçada",
                "kicking": "Chute",
                "running": "Correndo",
                "fighting": "Briga Detectada"
            }
            
            # Pega a tradução ou usa o original se não tiver na lista
            label = traducoes.get(label_ingles, label_ingles)
            # --------------------------------------------------------

            # Listas de palavras-chave (agora usando os nomes EM INGLÊS para lógica interna)
            violencia_keywords = ["fight", "punch", "kick", "hit", "boxing", "slap", "headbutt", "wrestling", "beating", "smacking", "striking"]
            suspeito_keywords = ["running", "jumping", "falling", "climbing"]
            ilicita_keywords = ["robbery", "burglary", "stealing"]

            print(f"\n{'='*60}")
            print(f"🔎 DETECÇÃO VideoMAE [{self.camera_nome}]")
            print(f"{'='*60}")
            print(f"   Label Original: '{label_ingles}'")
            print(f"   Label Exibido: '{label}'")
            print(f"   Confiança: {confianca:.1%}")

            evento = None
            # A verificação continua sendo feita no label original em inglês para garantir precisão
            if any(palavra in label_ingles for palavra in violencia_keywords):
                evento = "violencia_detectada"
            elif any(palavra in label_ingles for palavra in suspeito_keywords):
                evento = "comportamento_suspeito"
            elif any(palavra in label_ingles for palavra in ilicita_keywords):
                evento = "atividade_ilicita"

            # Força o evento se for beatboxing ou punching bag (caso as keywords não peguem)
            if label_ingles in ["beatboxing", "punching bag"]:
                 evento = "violencia_detectada"

            if evento:
                self.ultima_deteccao = {
                    'acao': label, # Salva o nome TRADUZIDO/PERSONALIZADO
                    'evento': evento,
                    'timestamp': datetime.now()
                }
                print(f"🚨 EVENTO CLASSIFICADO: {evento} ({label})")
                # Passa o label traduzido para a gravação
                self.iniciar_gravacao(frame_atual, evento, acao_detectada=label)
            else:
                print(f"ℹ️ Ação detectada mas não classificada como anomalia")

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
        """YOLO - configurável para CPU/GPU"""
        confianca_min = self.config.get('deteccao', 'confianca_minima', default=0.5)
        # Passar device explicitamente para evitar erro de meta tensor
        results = self.yolo(frame, verbose=False, conf=confianca_min, imgsz=320, device=self.yolo_device)

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
        # VideoMAE espera EXATAMENTE 16 frames
        # Se temos mais frames, pegamos frames espaçados uniformemente
        num_frames_necessarios = 16
        num_frames_disponiveis = len(video_clip)

        if num_frames_disponiveis == num_frames_necessarios:
            # Tamanho perfeito, usar todos
            frames_selecionados = video_clip
        elif num_frames_disponiveis > num_frames_necessarios:
            # Temos mais frames, selecionar uniformemente
            indices = [int(i * num_frames_disponiveis / num_frames_necessarios)
                      for i in range(num_frames_necessarios)]
            frames_selecionados = [video_clip[i] for i in indices]
        else:
            # Temos menos frames, duplicar o último
            frames_selecionados = list(video_clip)
            while len(frames_selecionados) < num_frames_necessarios:
                frames_selecionados.append(video_clip[-1])

        video_clip_otimizado = []
        for frame in frames_selecionados:
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

    # ==================== SISTEMA DUAL DE GRAVAÇÃO ====================

    def iniciar_gravacao_continua(self):
        """Inicia gravação contínua da sessão completa"""
        try:
            self.inicio_sessao = datetime.now()
            timestamp_str = self.inicio_sessao.strftime('%Y%m%d_%H%M%S')
            self.nome_video_continuo = f"sessao_{timestamp_str}.mp4"
            caminho_completo = os.path.join(self.pasta_sessoes, self.nome_video_continuo)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
            self.gravador_continuo = cv2.VideoWriter(caminho_completo, fourcc, float(self.fps_camera), (640, 480))

            if self.gravador_continuo.isOpened():
                self.gravacao_continua = True
                self.anomalias_detectadas = []

                self.arquivo_indice = os.path.join(
                    self.pasta_sessoes,
                    f"indice_{timestamp_str}.json"
                )

                print(f"📹 Gravação contínua iniciada: {self.nome_video_continuo}")

        except Exception as e:
            print(f"❌ Erro ao iniciar gravação contínua: {e}")

    def finalizar_gravacao_continua(self):
        """Finaliza gravação contínua e salva índice"""
        if self.gravador_continuo and self.gravacao_continua:
            try:
                self.gravador_continuo.release()
                self.gravador_continuo = None
                self.gravacao_continua = False

                # Salvar índice JSON
                if self.arquivo_indice and self.anomalias_detectadas:
                    duracao_total = (datetime.now() - self.inicio_sessao).total_seconds()

                    dados_indice = {
                        'video_sessao': self.nome_video_continuo,
                        'inicio_sessao': self.inicio_sessao.strftime('%Y-%m-%d %H:%M:%S'),
                        'duracao_total_segundos': duracao_total,
                        'total_anomalias': len(self.anomalias_detectadas),
                        'anomalias': self.anomalias_detectadas
                    }

                    with open(self.arquivo_indice, 'w', encoding='utf-8') as f:
                        json.dump(dados_indice, f, indent=4, ensure_ascii=False)

                    print(f"✅ Sessão finalizada: {len(self.anomalias_detectadas)} anomalias")

            except Exception as e:
                print(f"❌ Erro ao finalizar gravação contínua: {e}")

    def registrar_anomalia_no_indice(self, timestamp_deteccao, evento):
        """Registra anomalia no índice"""
        if not self.gravacao_continua or not self.inicio_sessao:
            return

        tempo_no_video = (timestamp_deteccao - self.inicio_sessao).total_seconds()

        anomalia_info = {
            'timestamp_absoluto': timestamp_deteccao.strftime('%Y-%m-%d %H:%M:%S'),
            'tempo_no_video_segundos': round(tempo_no_video, 2),
            'tempo_no_video_formatado': self.formatar_timestamp(tempo_no_video),
            'tipo': evento,
            'clip_associado': f"{evento}_{timestamp_deteccao.strftime('%Y%m%d_%H%M%S')}.mp4"
        }

        self.anomalias_detectadas.append(anomalia_info)
        print(f"📌 Anomalia registrada em {anomalia_info['tempo_no_video_formatado']}")

    def formatar_timestamp(self, segundos):
        """Formata segundos em HH:MM:SS"""
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        segs = int(segundos % 60)
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"

    def iniciar_gravacao(self, frame, evento_detectado, acao_detectada="desconhecida"):
        """Inicia gravação de CLIP (10s)"""
        if self.gravando:
            return

        altura, largura = frame.shape[:2]
        timestamp = datetime.now()
        # Nome do arquivo continua usando o evento genérico para facilitar organização
        nome = f"{evento_detectado}_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        caminho = os.path.join(self.pasta_videos, nome)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.video_writer = cv2.VideoWriter(caminho, fourcc, float(self.fps_camera), (largura, altura))

        self.gravando = True
        self.inicio_gravacao = time.time()

        # --- AQUI ESTÁ O SEGREDO ---
        # Adicionar ao histórico COM A TRADUÇÃO
        self.historico_videos.append({
            'nome': nome,
            'caminho': caminho,
            'evento': evento_detectado, # Categoria (ex: violencia_detectada)
            'acao': acao_detectada,     # Ação Traduzida (ex: Movimento Brusco) <-- OBRIGATÓRIO
            'timestamp': timestamp.strftime('%d/%m/%Y %H:%M:%S')
        })

        # Registrar no índice da sessão contínua
        self.registrar_anomalia_no_indice(timestamp, evento_detectado)

        print(f"🎥 Gravando CLIP: {nome} (Ação: {acao_detectada})")

    def processar_frame(self):
        """Processamento otimizado - YOLO na CPU"""
        if not self.rodando:
            return None

        # Limpa buffer da câmera
        for _ in range(2):
            ret, frame_atual = self.cap.read()
            if not ret:
                return None

        # NOVO: Gravar no vídeo contínuo
        if self.gravacao_continua and self.gravador_continuo:
            try:
                self.gravador_continuo.write(frame_atual)
            except Exception as e:
                print(f"❌ Erro ao gravar frame contínuo: {e}")

        # YOLO na CPU
        self.contador_frames += 1
        if self.contador_frames % self.intervalo_yolo == 0:
            self.num_pessoas, self.boxes_yolo = self.processar_yolo_rapido(frame_atual)

        # Desenha boxes
        if self.boxes_yolo:
            self.desenhar_deteccoes(frame_atual)

        # Informações visuais
        self.adicionar_info_visual(frame_atual)

        # Detecção de movimento
        if self.frame_anterior is not None:
            tempo_atual = time.time()
            if tempo_atual - self.ultimo_movimento_time > self.movimento_cooldown:
                movimento = self.detectar_movimento(self.frame_anterior, frame_atual)

                # Verificar se buffer está cheio (usa tamanho configurado)
                buffer_cheio = len(self.frame_buffer) == self.frame_buffer.maxlen

                if movimento and not self.gravando and buffer_cheio and not self.analisando:
                    self.analisando = True
                    self.ultimo_movimento_time = tempo_atual

                    video_clip_copy = list(self.frame_buffer)
                    frame_copy = frame_atual.copy()

                    try:
                        self.fila_analise.put_nowait((video_clip_copy, frame_copy))
                    except:
                        self.analisando = False

        self.frame_buffer.append(frame_atual)

        # Gravação de CLIP
        if self.gravando:
            self.video_writer.write(frame_atual)
            if time.time() - self.inicio_gravacao >= self.duracao_gravacao:
                self.video_writer.release()
                self.gravando = False
                print("💾 CLIP finalizado")

        self.frame_anterior = frame_atual.copy()
        self.ultimo_frame = frame_atual

        return frame_atual

    def adicionar_info_visual(self, frame):
        """Adiciona informações visuais no frame"""
        # Pessoas detectadas
        cv2.putText(frame, f"P: {self.num_pessoas}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Status de análise
        if self.analisando:
            cv2.putText(frame, "ANALISANDO", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Gravação contínua
        if self.gravacao_continua:
            duracao = (datetime.now() - self.inicio_sessao).total_seconds()
            tempo_formatado = self.formatar_timestamp(duracao)
            cv2.circle(frame, (30, 80), 8, (255, 0, 0), -1)
            cv2.putText(frame, f"REC SESSAO {tempo_formatado}", (50, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Gravação de clip
        if self.gravando:
            tempo_restante = int(self.duracao_gravacao - (time.time() - self.inicio_gravacao))
            cv2.circle(frame, (30, 105), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"CLIP {tempo_restante}s", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Contador de anomalias
        if len(self.anomalias_detectadas) > 0:
            cv2.putText(frame, f"Anomalias: {len(self.anomalias_detectadas)}", (450, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    def iniciar(self):
        """Inicia o sistema"""
        print("🔵 Iniciando sistema...")

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)

        self.rodando = True

        # NOVO: Iniciar gravação contínua
        self.iniciar_gravacao_continua()

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

        # Finalizar gravação de clip
        if self.gravando and self.video_writer:
            self.video_writer.release()
            self.gravando = False

        # NOVO: Finalizar gravação contínua
        if self.gravacao_continua:
            self.finalizar_gravacao_continua()

        if self.cap.isOpened():
            self.cap.release()

    def get_historico_videos(self):
        """Retorna lista de vídeos"""
        return sorted(self.historico_videos, key=lambda x: x['timestamp'], reverse=True)

    def get_indices_disponiveis(self):
        """Retorna lista de arquivos de índice disponíveis"""
        import glob
        indices = glob.glob(os.path.join(self.pasta_sessoes, 'indice_*.json'))
        return sorted(indices, key=os.path.getctime, reverse=True)