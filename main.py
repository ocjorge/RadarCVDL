import cv2
import torch
import numpy as np
import time
import json
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt

try:
    from torchvision.ops import nms
except ImportError:
    print("Advertencia: torchvision no encontrado. La supresión de no máximos (NMS) no funcionará.")
    print("Instálalo con: pip install torchvision")
    nms = None


class RiskRadarSystem:
    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.model_path_vehicles = config['MODEL_PATH_VEHICLES']
        self.video_path = config['VIDEO_INPUT_PATH']
        self.CONFIDENCE_THRESHOLD = config['YOLO_CONFIDENCE_THRESHOLD']
        self.NMS_IOU_THRESHOLD = config['NMS_IOU_THRESHOLD']
        self.PROCESS_EVERY_N_FRAMES = config['PROCESS_EVERY_N_FRAMES']
        os.makedirs(output_dir, exist_ok=True)
        self.detection_data, self.frame_stats, self.processing_times, self.risk_history = [], [], [], []
        self.active_trackers, self.next_tracker_id, self.last_depth_map = {}, 0, None
        self._load_models()
        self._setup_video()
        self._setup_risk_components()
        self._setup_logging()

    def _load_models(self):
        print("Cargando modelos...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        self.model_vehicles = YOLO(self.model_path_vehicles)
        self.model_coco = YOLO('yolov8n.pt')
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform
        print("Modelos cargados exitosamente.")

    def _setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened(): raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.proc_width, self.proc_height = self.config.get('PROCESSING_RESOLUTION',
                                                            (self.original_width, self.original_height))
        print(f"Resolución original: {self.original_width}x{self.original_height}")
        print(f"Procesando a resolución: {self.proc_width}x{self.proc_height}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(os.path.join(self.output_dir, 'output_risk_radar.mp4'), fourcc, self.fps,
                                         (self.original_width, self.original_height))

    def _setup_risk_components(self):
        # CORRECCIÓN 1: LÓGICA DEL CONO INVERTIDA
        heatmap_h = int(self.proc_height * self.config['HEATMAP_RESOLUTION_FACTOR'])
        heatmap_w = int(self.proc_width * self.config['HEATMAP_RESOLUTION_FACTOR'])
        self.risk_heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        self.cone_mask = np.zeros((self.proc_height, self.proc_width), dtype=np.uint8)
        bottom_center_x = self.proc_width // 2
        bottom_y = self.proc_height - 1
        top_width = self.proc_width * self.config['CONE_TOP_WIDTH_FACTOR']
        top_y = 0
        p1 = (bottom_center_x, bottom_y)
        p2 = (int(bottom_center_x + top_width / 2), top_y)
        p3 = (int(bottom_center_x - top_width / 2), top_y)
        cone_points = np.array([p1, p2, p3], np.int32)
        cv2.fillPoly(self.cone_mask, [cone_points], 255)
        self.cone_mask_low_res = cv2.resize(self.cone_mask, (heatmap_w, heatmap_h), interpolation=cv2.INTER_NEAREST) > 0

    def _setup_logging(self):
        self.log_file = os.path.join(self.output_dir, 'processing_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Inicio del procesamiento: {datetime.now()}\n")
            config_str = {k: str(v) for k, v in self.config.items()}
            f.write(json.dumps(config_str, indent=4) + "\n")
            f.write("-" * 50 + "\n")

    def _log_message(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def _add_heat(self, center_x, center_y, radius, value):
        h, w = self.risk_heatmap.shape
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = dist_from_center <= radius
        self.risk_heatmap[mask] += value

    def _run_full_detection(self, frame):
        results_v = self.model_vehicles.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        results_c = self.model_coco.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD,
                                            classes=self.config['COCO_CLASSES_TO_SEEK_IDS'], verbose=False)
        all_boxes, all_scores, all_class_names = [], [], []
        detections_source = [(results_v[0].boxes, self.model_vehicles.names),
                             (results_c[0].boxes, self.model_coco.names)]
        for boxes, names_map in detections_source:
            if boxes:
                for box in boxes:
                    all_boxes.append(box.xyxy[0])
                    all_scores.append(box.conf[0])
                    all_class_names.append(names_map[int(box.cls[0])])
        detections_final = []
        if all_boxes and nms is not None:
            indices = nms(torch.stack(all_boxes), torch.stack(all_scores), self.NMS_IOU_THRESHOLD)
            self.active_trackers = {}
            for i in indices:
                box = all_boxes[i].cpu().numpy().astype(int)
                class_name = all_class_names[i]
                conf = all_scores[i].item()
                detections_final.append({'bbox': box, 'class_name': class_name, 'conf': conf})
                tracker = cv2.TrackerCSRT_create()
                x1, y1, x2, y2 = box
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                self.active_trackers[self.next_tracker_id] = {'tracker': tracker, 'class_name': class_name}
                self.next_tracker_id += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img_rgb.shape[:2],
                                                         mode="bicubic", align_corners=False).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = depth_map.min(), depth_map.max()
        self.last_depth_map = (depth_map - depth_min) / (
                    depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_map)
        return detections_final

    def _run_tracker_update(self, frame):
        if not self.active_trackers: return []
        detections_final, lost_trackers = [], []
        for tracker_id, data in self.active_trackers.items():
            success, bbox = data['tracker'].update(frame)
            if success:
                x1, y1, w, h = [int(v) for v in bbox]
                detections_final.append(
                    {'bbox': np.array([x1, y1, x1 + w, y1 + h]), 'class_name': data['class_name'], 'conf': 1.0})
            else:
                lost_trackers.append(tracker_id)
        for tracker_id in lost_trackers: del self.active_trackers[tracker_id]
        return detections_final

    def _update_risk_and_visualize(self, frame, detections, frame_number, timestamp):
        frame_start_time = time.time()
        self.risk_heatmap *= self.config['HEATMAP_DECAY_RATE']
        frame_detections_data = []
        if self.last_depth_map is not None:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # CORRECCIÓN 3: Convertir tipos a nativos de Python para evitar error JSON
                frame_detections_data.append({
                    'frame_number': int(frame_number), 'timestamp': float(timestamp), 'class': str(det['class_name']),
                    'confidence': float(det['conf']), 'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

                if 0 <= cy < self.proc_height and 0 <= cx < self.proc_width and self.cone_mask[cy, cx] == 255:
                    depth_roi = self.last_depth_map[y1:y2, x1:x2]
                    median_normalized_depth = np.median(depth_roi) if depth_roi.size > 0 else 0
                    base_heat = self.config['HEAT_INTENSITY_FACTORS'].get(det['class_name'], 0.3)

                    # CORRECCIÓN 2: LÓGICA DE RIESGO MEJORADA
                    depth_factor = 1 + median_normalized_depth
                    position_factor = 1 + (cy / self.proc_height)
                    heat_to_add = base_heat * depth_factor * position_factor

                    hm_cx = int(cx * self.config['HEATMAP_RESOLUTION_FACTOR'])
                    hm_cy = int(cy * self.config['HEATMAP_RESOLUTION_FACTOR'])
                    self._add_heat(center_x=hm_cx, center_y=hm_cy, radius=5, value=heat_to_add)

        self.detection_data.extend(frame_detections_data)
        total_heat_in_cone = np.sum(self.risk_heatmap[self.cone_mask_low_res])
        risk_level, risk_color = "Bajo", (0, 255, 0)
        if total_heat_in_cone > self.config['HEAT_THRESHOLD_HIGH']:
            risk_level, risk_color = "Alto", (0, 0, 255)
        elif total_heat_in_cone > self.config['HEAT_THRESHOLD_MEDIUM']:
            risk_level, risk_color = "Medio", (0, 165, 255)
        self.risk_history.append(risk_level)
        annotated_frame = self._visualize_frame(frame.copy(), detections, risk_level, risk_color, total_heat_in_cone)
        processing_time = time.time() - frame_start_time
        self.processing_times.append(processing_time)
        self.frame_stats.append(
            {'frame_number': frame_number, 'timestamp': timestamp, 'detection_count': len(detections),
             'processing_time': processing_time, 'total_heat': total_heat_in_cone, 'risk_level': risk_level})
        return annotated_frame

    def process_video(self):
        self._log_message("Iniciando procesamiento de video optimizado...")
        start_time = time.time()
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            proc_frame = cv2.resize(frame, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
            timestamp = frame_count / self.fps
            detections = self._run_full_detection(
                proc_frame) if frame_count % self.PROCESS_EVERY_N_FRAMES == 0 else self._run_tracker_update(proc_frame)
            annotated_proc_frame = self._update_risk_and_visualize(proc_frame, detections, frame_count, timestamp)
            annotated_original_frame = cv2.resize(annotated_proc_frame, (self.original_width, self.original_height),
                                                  interpolation=cv2.INTER_LINEAR)
            self.out_video.write(annotated_original_frame)
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                self._log_message(f"Procesados {frame_count}/{self.total_frames} frames | Velocidad: {avg_fps:.2f} FPS")
        self._log_message(f"Procesamiento completado en {time.time() - start_time:.1f}s")
        self._cleanup()
        self._generate_reports()

    def _visualize_frame(self, frame, detections, risk_level, risk_color, total_heat):
        vis_frame = frame
        h, w, _ = vis_frame.shape
        heatmap_upscaled = cv2.resize(self.risk_heatmap, (w, h))
        heatmap_normalized = cv2.normalize(heatmap_upscaled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        masked_heatmap = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=self.cone_mask)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, masked_heatmap, 0.5, 0)
        contours, _ = cv2.findContours(self.cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(vis_frame, contours, isClosed=True, color=(255, 255, 0), thickness=2)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name, conf = det['class_name'], det.get('conf', 1.0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.putText(vis_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
        cv2.rectangle(vis_frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"RIESGO: {risk_level.upper()}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 1, risk_color, 2)
        cv2.putText(vis_frame, f"Heat: {total_heat:.2f}", (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    1)
        return vis_frame

    def _cleanup(self):
        self.cap.release()
        self.out_video.release()
        cv2.destroyAllWindows()
        self._log_message("Recursos liberados.")

    def _generate_reports(self):
        self._log_message("Generando reportes finales...")
        if self.detection_data:
            with open(os.path.join(self.output_dir, 'detections_raw.json'), 'w') as f:
                json.dump(self.detection_data, f, indent=2)
            with open(os.path.join(self.output_dir, 'detections_raw.csv'), 'w', newline='') as f:
                csv_data = [
                    {'frame_number': d['frame_number'], 'timestamp': f"{d['timestamp']:.2f}", 'class': d['class'],
                     'confidence': f"{d['confidence']:.2f}", 'x1': d['bbox'][0], 'y1': d['bbox'][1],
                     'x2': d['bbox'][2], 'y2': d['bbox'][3]} for d in self.detection_data]
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
        self._generate_statistics_report()
        self._generate_visualizations()
        self._log_message("Reportes generados exitosamente.")

    def _generate_statistics_report(self):
        stats = {
            'video_info': {'path': self.video_path, 'resolution': f"{self.original_width}x{self.original_height}",
                           'processing_resolution': f"{self.proc_width}x{self.proc_height}", 'fps': self.fps,
                           'total_frames': self.total_frames},
            'processing_info': {'frames_processed': len(self.frame_stats), 'avg_processing_time_per_frame': np.mean(
                self.processing_times) if self.processing_times else 0,
                                'total_processing_time': sum(self.processing_times)},
            'detection_stats': {}, 'risk_analysis': {}
        }
        if self.detection_data: stats['detection_stats']['by_class'] = dict(
            Counter([d['class'] for d in self.detection_data]))
        if self.risk_history:
            risk_counts = Counter(self.risk_history)
            total_risk_frames = len(self.risk_history)
            stats['risk_analysis']['time_in_risk_level_percent'] = {level: (count / total_risk_frames) * 100 for
                                                                    level, count in risk_counts.items()}
            stats['risk_analysis']['risk_level_counts'] = dict(risk_counts)
        with open(os.path.join(self.output_dir, 'statistics_report.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("REPORTE DE ANÁLISIS DE RIESGO\n" + "=" * 50 + "\n\n")
            f.write("ANÁLISIS DE RIESGO:\n")
            for level, perc in stats['risk_analysis'].get('time_in_risk_level_percent', {}).items(): f.write(
                f"  - Tiempo en Riesgo '{level}': {perc:.2f}%\n")
            f.write("\nDETECCIONES POR CLASE:\n")
            for class_name, count in stats['detection_stats'].get('by_class', {}).items(): f.write(
                f"  - {class_name}: {count}\n")
            f.write(f"\nPROCESAMIENTO:\n  - Frames procesados: {stats['processing_info']['frames_processed']}\n")
            f.write(
                f"  - Tiempo promedio por frame: {stats['processing_info']['avg_processing_time_per_frame']:.3f}s\n")

    def _generate_visualizations(self):
        if not self.frame_stats: return
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Análisis del Procesamiento de Riesgo', fontsize=16)
        if self.detection_data:
            class_counts = Counter([d['class'] for d in self.detection_data])
            axes[0, 0].bar(class_counts.keys(), class_counts.values(), color='skyblue')
            axes[0, 0].set_title('Distribución de Detecciones por Clase')
            axes[0, 0].tick_params(axis='x', rotation=45)
        total_heat_history = [s['total_heat'] for s in self.frame_stats]
        axes[0, 1].plot(total_heat_history, color='orangered')
        axes[0, 1].set_title('Nivel de "Calor" en el Cono a lo Largo del Tiempo')
        axes[0, 1].set_xlabel('Número de Frame');
        axes[0, 1].set_ylabel('Calor Total')
        if self.risk_history:
            risk_color_map = {'Bajo': '#2ca02c', 'Medio': '#ff7f0e', 'Alto': '#d62728'}
            risk_counts = Counter(self.risk_history)
            sorted_keys = sorted(risk_counts.keys(), key=lambda x: list(risk_color_map.keys()).index(x))
            axes[1, 0].pie([risk_counts[key] for key in sorted_keys], labels=sorted_keys, autopct='%1.1f%%',
                           colors=[risk_color_map[key] for key in sorted_keys], startangle=90)
            axes[1, 0].set_title('Distribución de Tiempo por Nivel de Riesgo');
            axes[1, 0].axis('equal')
        axes[1, 1].plot(self.processing_times, color='purple', alpha=0.7)
        axes[1, 1].set_title('Tiempo de Procesamiento por Frame')
        axes[1, 1].set_xlabel('Número de Frame');
        axes[1, 1].set_ylabel('Segundos')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'analysis_charts.png'), dpi=300)
        plt.close()


def main():
    config = {
        'MODEL_PATH_VEHICLES': 'F:/Documents/PycharmProjects/DepthDetector/best.pt',
        'VIDEO_INPUT_PATH': 'F:\Documents\PycharmProjects\RiskRadarOptimized\GH012372_no_audio.mp4',
        'OUTPUT_DIR': 'results_risk_radar_optimized',
        'PROCESS_EVERY_N_FRAMES': 5,
        'PROCESSING_RESOLUTION': (960, 540),
        'NMS_IOU_THRESHOLD': 0.6,
        'YOLO_CONFIDENCE_THRESHOLD': 0.40,
        'COCO_CLASSES_TO_SEEK_IDS': [0, 1, 16],
        'CONE_BOTTOM_Y_FACTOR': 0.95,  # Este ya no se usa para la forma del cono, pero puede ser útil si lo reutilizas.
        'CONE_TOP_WIDTH_FACTOR': 0.8,
        'HEATMAP_RESOLUTION_FACTOR': 0.25,
        'HEATMAP_DECAY_RATE': 0.85,
        'HEAT_INTENSITY_FACTORS': {'car': 0.8, 'threewheel': 0.7, 'bus': 1.0, 'truck': 1.0, 'motorbike': 0.6,
                                   'van': 0.9, 'person': 0.4, 'bicycle': 0.3, 'dog': 0.5},
        'HEAT_THRESHOLD_MEDIUM': 25.0,  # Ajustado ligeramente hacia arriba por el nuevo factor de posición
        'HEAT_THRESHOLD_HIGH': 50.0,  # Ajustado ligeramente hacia arriba por el nuevo factor de posición
    }
    if not os.path.exists(config['MODEL_PATH_VEHICLES']):
        print(f"Error: No se encontró el modelo en {config['MODEL_PATH_VEHICLES']}")
        return
    if not os.path.exists(config['VIDEO_INPUT_PATH']):
        print(f"Error: No se encontró el video en {config['VIDEO_INPUT_PATH']}")
        return
    try:
        radar = RiskRadarSystem(config['OUTPUT_DIR'], config)
        radar.process_video()
        print(f"\n✅ Procesamiento optimizado completado exitosamente!")
        print(f"📁 Resultados guardados en: {os.path.abspath(config['OUTPUT_DIR'])}/")
    except Exception as e:
        print(f"❌ Error catastrófico durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
