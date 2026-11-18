import os
import cv2
import time
import joblib
import threading
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gpiozero import DistanceSensor

# =============================================================================
# CONFIGURACIÓN (ajusta aquí lo necesario para tu Raspberry)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # funciona también al arrancar en autoinicio
CARPETA_AUDIOS = os.path.join(BASE_DIR, "audios")
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_random_forest_2manos.pkl")
RUTA_HAND_TASK = os.path.join(BASE_DIR, "hand_landmarker.task")

# --- Sensor ultrasónico HC-SR04 (numeración BCM, no pin físico) ---
PIN_TRIG = 23                 # pin físico 16
PIN_ECHO = 24                 # pin físico 18 (¡con divisor de voltaje 5V -> 3.3V!)
DISTANCIA_ACTIVACION_M = 0.80 # se enciende si hay alguien a menos de 80 cm
SEG_PRESENCIA_CONFIRMAR = 0.6 # presencia sostenida para encender
SEG_AUSENCIA_APAGAR = 10.0    # sin presencia ni manos este tiempo -> vuelve a reposo

# --- Cámara / rendimiento ---
ANCHO, ALTO = 640, 480
PROCESAR_CADA_N_FRAMES = 2    # sube a 3 si va lento en tu modelo de Raspberry
MOSTRAR_VENTANA = bool(os.environ.get("DISPLAY"))   # sin monitor -> modo headless automático

# --- Lógica del traductor ---
TIEMPO_CONFIRMAR_SENIA = 2.0
TIEMPO_PAUSA_PALABRA = 3.0
UMBRAL_CONFIANZA = 0.85
NOMBRE_VENTANA = "Traductor - Frase Completa"

# =============================================================================
# 1. AUDIO (pygame con buffer mayor para evitar cortes en Raspberry)
# =============================================================================
AUDIO_OK = False
try:
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=2048)
    pygame.mixer.init()
    AUDIO_OK = True
except pygame.error as e:
    print(f"[AUDIO] No se pudo iniciar el mixer: {e}")
    print("[AUDIO] Revisa la salida de audio (raspi-config / alsamixer) o conecta bocina/parlante.")

_reproduciendo_cola = threading.Event()


def buscar_archivo_audio(nombre_base: str):
    """Busca el archivo en 'audios/' aceptando .wav, .mp3 u .ogg (nombre en mayúsculas)."""
    limpio = nombre_base.strip().upper()
    for ext in (".wav", ".mp3", ".ogg"):
        ruta = os.path.join(CARPETA_AUDIOS, f"{limpio}{ext}")
        if os.path.exists(ruta):
            return ruta
    return None


def reproducir_archivo_bloqueante(ruta: str):
    pygame.mixer.music.load(ruta)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)


def _reproducir_secuencia_archivos(lista_rutas: list):
    try:
        for ruta in lista_rutas:
            if ruta and os.path.exists(ruta):
                reproducir_archivo_bloqueante(ruta)
                time.sleep(0.25)
            else:
                print(f"[AUDIO] No se encontró el archivo: '{ruta}'")
    except Exception as e:
        print(f"[AUDIO] Error en reproducción: {e}")
    finally:
        _reproduciendo_cola.clear()


def reproducir_bufer_audio_async(lista_rutas: list):
    if not AUDIO_OK:
        print("[AUDIO] Mixer no disponible, no se reproduce nada.")
        return
    if not _reproduciendo_cola.is_set() and lista_rutas:
        _reproduciendo_cola.set()
        threading.Thread(target=_reproducir_secuencia_archivos,
                         args=(lista_rutas.copy(),), daemon=True).start()


def audio_en_curso() -> bool:
    return _reproduciendo_cola.is_set()


# =============================================================================
# 2. SENSOR ULTRASÓNICO
# =============================================================================
sensor = DistanceSensor(echo=PIN_ECHO, trigger=PIN_TRIG,
                        max_distance=2.0, queue_len=5)


def hay_presencia() -> bool:
    try:
        return sensor.distance < DISTANCIA_ACTIVACION_M
    except Exception:
        return False


def esperar_presencia():
    """Bloquea (consumo mínimo) hasta que haya alguien cerca de forma sostenida."""
    print(f"\n[REPOSO] Esperando persona a menos de {DISTANCIA_ACTIVACION_M*100:.0f} cm...")
    desde = None
    while True:
        if hay_presencia():
            desde = desde or time.time()
            if time.time() - desde >= SEG_PRESENCIA_CONFIRMAR:
                return
        else:
            desde = None
        time.sleep(0.1)


# =============================================================================
# 3. CÁMARA (webcam USB o módulo de cámara de Raspberry con picamera2)
# =============================================================================
class Camara:
    def __init__(self):
        self.cap = None
        self.picam = None
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
        if cap.isOpened():
            self.cap = cap
            return
        cap.release()
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            self.picam.configure(self.picam.create_preview_configuration(
                main={"format": "RGB888", "size": (ANCHO, ALTO)}))
            self.picam.start()
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir ninguna cámara: {e}")

    def read(self):
        if self.cap is not None:
            return self.cap.read()
        # picamera2 con formato RGB888 entrega el arreglo en orden BGR (compatible con OpenCV)
        return True, self.picam.capture_array()

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()


# =============================================================================
# 4. MODELO Y DETECTOR
# =============================================================================
print("Cargando modelo Random Forest (1 y 2 manos)...")
modelo = joblib.load(RUTA_MODELO)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=RUTA_HAND_TASK),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


# =============================================================================
# 5. SESIÓN DE TRADUCCIÓN (se ejecuta solo mientras hay alguien presente)
# =============================================================================
def sesion_traductor():
    palabra_actual = ""
    frase_pantalla = ""
    rutas_letras_actuales = []
    bufer_rutas_audio = []

    senia_actual_estable = ""
    tiempo_inicio_senia = 0.0
    tiempo_ultima_deteccion = time.time()
    ultima_actividad = time.time()   # última vez que hubo presencia o manos o audio
    contador_frames = 0

    cam = Camara()
    print("\n=== TRADUCTOR ACTIVADO ===")
    if MOSTRAR_VENTANA:
        print(" 'q'/ESC: salir | 'c': borrar | ESPACIO: reproducir ahora")

    # Saludo opcional si existe audios/BIENVENIDA.wav|mp3
    saludo = buscar_archivo_audio("BIENVENIDA")
    if saludo:
        reproducir_bufer_audio_async([saludo])

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            tiempo_actual = time.time()
            contador_frames += 1
            flip_frame = cv2.flip(frame, 1)
            h, w, _ = flip_frame.shape

            se_detecto_mano_valida = False
            manos_en_pantalla = 0
            texto_estado_inferior = ""
            color_estado_inferior = (200, 200, 200)

            # Ahorro de CPU: procesar solo 1 de cada N fotogramas
            if contador_frames % PROCESAR_CADA_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    manos_en_pantalla = len(detection_result.hand_landmarks)
                    vector_izq = np.zeros(63, dtype=np.float32)
                    vector_der = np.zeros(63, dtype=np.float32)
                    izq_ocupada = False
                    der_ocupada = False
                    wrist_ref = None

                    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        if MOSTRAR_VENTANA:
                            for lm in hand_landmarks:
                                cv2.circle(flip_frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

                        if wrist_ref is None:
                            wrist_ref = (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h))

                        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                        landmarks_norm = (raw - raw[0]).flatten()

                        lateralidad = detection_result.handedness[idx][0].category_name
                        if lateralidad == "Left" and not izq_ocupada:
                            vector_izq = landmarks_norm
                            izq_ocupada = True
                        elif not der_ocupada:
                            vector_der = landmarks_norm
                            der_ocupada = True
                        else:
                            vector_izq = landmarks_norm
                            izq_ocupada = True

                    vector_126 = np.concatenate([vector_izq, vector_der]).reshape(1, -1)
                    # Una sola inferencia (antes eran dos: predict + predict_proba)
                    probas = modelo.predict_proba(vector_126)[0]
                    i_max = int(np.argmax(probas))
                    prediccion = str(modelo.classes_[i_max])
                    confianza = float(probas[i_max])

                    if confianza >= UMBRAL_CONFIANZA:
                        se_detecto_mano_valida = True
                        tiempo_ultima_deteccion = tiempo_actual

                        if prediccion != senia_actual_estable:
                            senia_actual_estable = prediccion
                            tiempo_inicio_senia = tiempo_actual

                        tiempo_transcurrido = tiempo_actual - tiempo_inicio_senia
                        progreso_pct = min(100, int((tiempo_transcurrido / TIEMPO_CONFIRMAR_SENIA) * 100))

                        if wrist_ref is not None and MOSTRAR_VENTANA:
                            cv2.putText(flip_frame, f"{prediccion} ({progreso_pct}%)",
                                        (wrist_ref[0] - 30, wrist_ref[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        texto_estado_inferior = (f"Confirmando '{prediccion}': "
                                                 f"{min(tiempo_transcurrido, TIEMPO_CONFIRMAR_SENIA):.1f}s / "
                                                 f"{TIEMPO_CONFIRMAR_SENIA:.1f}s")
                        color_estado_inferior = (0, 255, 0)

                        if tiempo_transcurrido >= TIEMPO_CONFIRMAR_SENIA:
                            if len(prediccion) > 1:
                                if palabra_actual != "":
                                    audio_palabra = buscar_archivo_audio(palabra_actual)
                                    if audio_palabra:
                                        bufer_rutas_audio.append(audio_palabra)
                                    else:
                                        bufer_rutas_audio.extend(rutas_letras_actuales)
                                    frase_pantalla += palabra_actual + " "
                                    palabra_actual = ""
                                    rutas_letras_actuales = []

                                frase_pantalla += f"[{prediccion}] "
                                ruta_senia = buscar_archivo_audio(prediccion)
                                if ruta_senia:
                                    bufer_rutas_audio.append(ruta_senia)
                                else:
                                    print(f"[AUDIO] Falta audio de '{prediccion}' en '{CARPETA_AUDIOS}/'")
                            else:
                                palabra_actual += prediccion
                                ruta_letra = buscar_archivo_audio(prediccion)
                                if ruta_letra:
                                    rutas_letras_actuales.append(ruta_letra)
                                else:
                                    print(f"[AUDIO] Falta audio de '{prediccion}' en '{CARPETA_AUDIOS}/'")

                            print(f"-> Confirmado: '{prediccion}' | Frase: '{frase_pantalla + palabra_actual}'")
                            tiempo_inicio_senia = tiempo_actual
                    else:
                        senia_actual_estable = ""
                        if wrist_ref is not None and MOSTRAR_VENTANA:
                            cv2.putText(flip_frame, "Analizando...",
                                        (wrist_ref[0] - 30, wrist_ref[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                if not se_detecto_mano_valida:
                    senia_actual_estable = ""

            # Actividad: presencia por sensor, manos visibles o audio sonando
            if hay_presencia() or manos_en_pantalla > 0 or audio_en_curso():
                ultima_actividad = tiempo_actual

            # --- Temporizador de fin de frase ---
            tiempo_sin_mano = tiempo_actual - tiempo_ultima_deteccion
            hay_pendientes = bool(bufer_rutas_audio or rutas_letras_actuales or palabra_actual)

            if not se_detecto_mano_valida and hay_pendientes:
                restante = max(0.0, TIEMPO_PAUSA_PALABRA - tiempo_sin_mano)
                texto_estado_inferior = f"Sin manos - Reproduciendo y limpiando en: {restante:.1f}s"
                color_estado_inferior = (0, 165, 255)
            elif not se_detecto_mano_valida and not texto_estado_inferior:
                texto_estado_inferior = "Esperando senia..."

            if hay_pendientes and tiempo_sin_mano > TIEMPO_PAUSA_PALABRA:
                if palabra_actual != "":
                    audio_palabra = buscar_archivo_audio(palabra_actual)
                    if audio_palabra:
                        bufer_rutas_audio.append(audio_palabra)
                    else:
                        bufer_rutas_audio.extend(rutas_letras_actuales)
                    frase_pantalla += palabra_actual

                print(f"\nFrase finalizada: '{frase_pantalla.strip()}'")
                print(f"Reproduciendo {len(bufer_rutas_audio)} archivo(s)...")
                reproducir_bufer_audio_async(bufer_rutas_audio)

                frase_pantalla = ""
                palabra_actual = ""
                rutas_letras_actuales = []
                bufer_rutas_audio = []
                senia_actual_estable = ""

            # --- Apagado por ausencia ---
            if (tiempo_actual - ultima_actividad) > SEG_AUSENCIA_APAGAR and not hay_pendientes:
                print("[INFO] Sin presencia. Volviendo a reposo.")
                return True

            # --- Interfaz y teclas (solo si hay monitor) ---
            if MOSTRAR_VENTANA:
                cv2.putText(flip_frame, f"Frase: {frase_pantalla + palabra_actual}", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
                cv2.putText(flip_frame, texto_estado_inferior, (20, ALTO - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_estado_inferior, 2)
                cv2.putText(flip_frame, "[Q/ESC] Salir | [C] Borrar | [ESPACIO] Hablar ahora",
                            (20, ALTO - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(NOMBRE_VENTANA, flip_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    return False   # salir del programa completo
                elif key in (ord('c'), ord('C')):
                    frase_pantalla = ""
                    palabra_actual = ""
                    rutas_letras_actuales = []
                    bufer_rutas_audio = []
                    senia_actual_estable = ""
                    print("[INFO] Búfer limpiado manualmente.")
                elif key == ord(' ') and hay_pendientes:
                    tiempo_ultima_deteccion = tiempo_actual - (TIEMPO_PAUSA_PALABRA + 0.1)

                if cv2.getWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                    return False
            else:
                time.sleep(0.005)
    finally:
        cam.release()
        if MOSTRAR_VENTANA:
            cv2.destroyAllWindows()
    return True


# =============================================================================
# 6. BUCLE PRINCIPAL: REPOSO -> (sensor) -> TRADUCTOR -> REPOSO
# =============================================================================
if __name__ == "__main__":
    print("Sistema listo. Modo ventana:", "SI" if MOSTRAR_VENTANA else "NO (headless)")
    try:
        continuar = True
        while continuar:
            esperar_presencia()
            continuar = sesion_traductor()
    except KeyboardInterrupt:
        print("\nSaliendo...")
    finally:
        sensor.close()
        if AUDIO_OK:
            pygame.mixer.quit()