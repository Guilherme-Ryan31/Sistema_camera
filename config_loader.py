import json
import os


class ConfigLoader:
    """Classe para carregar e gerenciar configurações"""

    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Carrega configurações do arquivo JSON"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ Configurações carregadas de {self.config_path}")
                return config
            else:
                print(f"⚠️ Arquivo {self.config_path} não encontrado, usando padrões")
                return self.get_default_config()
        except Exception as e:
            print(f"❌ Erro ao carregar configurações: {e}")
            return self.get_default_config()

    def get_default_config(self):
        """Retorna configurações padrão"""
        return {
            "sistema": {
                "usar_gpu": True,
                "fps_camera": 30,
                "resolucao": {"largura": 640, "altura": 480},
                "buffer_frames": 16
            },
            "cameras": [
                {"id": 0, "nome": "Câmera Principal", "source": 0, "ativa": True, "tipo": "webcam"}
            ],
            "deteccao": {
                "modelo_yolo": "yolov8n.pt",
                "modelo_videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
                "confianca_minima": 0.5,
                "intervalo_analise": 16
            },
            "gravacao": {
                "qualidade_jpeg": 70,
                "duracao_clip_anomalia": 10,
                "pasta_videos_sessoes": "videos_sessoes",
                "pasta_videos_anomalias": "videos_anomalias"
            },
            "servidor": {
                "host": "0.0.0.0",
                "porta": 5000,
                "debug": True
            }
        }

    def get(self, *keys, default=None):
        """Acessa valores aninhados com segurança"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def save_config(self):
        """Salva configurações atuais no arquivo"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configurações salvas em {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar configurações: {e}")
            return False

    def add_camera(self, nome, source, tipo="webcam"):
        """Adiciona nova câmera à configuração"""
        cameras = self.config.get('cameras', [])
        novo_id = max([c['id'] for c in cameras]) + 1 if cameras else 0

        nova_camera = {
            "id": novo_id,
            "nome": nome,
            "source": source,
            "ativa": True,
            "tipo": tipo
        }

        cameras.append(nova_camera)
        self.config['cameras'] = cameras
        self.save_config()

        return novo_id

    def get_cameras_ativas(self):
        """Retorna lista de câmeras ativas"""
        cameras = self.config.get('cameras', [])
        return [c for c in cameras if c.get('ativa', True)]

    def toggle_camera(self, camera_id, ativa):
        #Ativa/desativa uma câmera
        cameras = self.config.get('cameras', [])
        for camera in cameras:
            if camera['id'] == camera_id:
                camera['ativa'] = ativa
                self.save_config()
                return True
        return False
